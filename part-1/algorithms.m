%
% algorithms.m
%
% Created on May 17, 2017.
%

clc;

for k = 1:15
   
    filename = sprintf('es413-images/%d.jpg', k);
    sobelFilename = sprintf('es413-images/%d-sobel.jpg', k);
    laplacianFilename = sprintf('es413-images/%d-laplacian.jpg', k);
    cannyFilename = sprintf('es413-images/%d-canny.jpg', k);
    
    image = imread(filename);
    
    tImage = sobel(image, 3, 2);
    imwrite(tImage, sobelFilename);
    
    tImage = laplacian(image, 3, 5, 1);
    imwrite(tImage, laplacianFilename);
    
    tImage = canny_edges(image);
    imwrite(tImage, cannyFilename);
    
end

%
% Helpers.

function mask = laplacianKernel(n)

    mask = ones(n);
    mask(ceil((n^2)/2)) = 1 - n^2;
    
end

function k = rootMeanSquare(imageA, imageB)

    imageSize = size(imageA);
    imageWidth = imageSize(2);
    imageHeight = imageSize(1);
    
    k = imageA;
    
    for i = 2:(imageHeight-1)
        
        for j = 2:(imageWidth-1)
            
            a = double(imageA(i, j));
            b = double(imageB(i, j));
            k(i, j) = hypot(a, b);
        
        end
        
    end
    
end

function k = generateGaussianKenel(dimension, deviation)

    k = zeros(dimension, dimension);
    
    c1 = ((dimension-1)/2);
    c2 = (2*(deviation^2));
    c3 = (1/(2*pi*(deviation^2)));
    
    for i = 1:dimension
       
        for j = 1:dimension
            
            v1 = ((i-(c1+1))^2);
            v2 = ((j-(c1+1))^2);
            v3 = ((v1+v2)/c2);
            v4 = (c3*(1/exp(v3)));
            
            k(i, j) = v4;
            
        end
        
    end

end

function k = grayscaleImage(image)

    weights = [0.299, 0.587, 0.114];

    imageSize = size(image);
    imageWidth = imageSize(2);
    imageHeight = imageSize(1);
    
    k = zeros(imageHeight, imageWidth, 'double');
    
    for i = 1:imageHeight
        
       for j = 1:imageWidth
           
           for g = 1:3 
               
               k(i, j) = k(i, j) + (image(i, j, g) * weights(g));
               
           end
           
       end
       
    end
    
end

function c = kernelCore(matrix, kernel, kSize)
    
    c = 0;
    
    for i = 1:kSize
        
        for j = 1:kSize
            
            a = matrix(i, j);
            b = kernel(i, j);
            c = c + (a * b);
            
        end
        
    end
    
end

function k = applyKernel(image, kernel, kSize)

    wSpan = floor(kSize/2);

    imageSize = size(image);
    imageWidth = imageSize(2);
    imageHeight = imageSize(1);
    
    k = zeros(imageHeight, imageWidth, 'double');
    
    for i = (wSpan+1):(imageHeight-wSpan-1)
        
       for j = (wSpan+1):(imageWidth-wSpan-1)
           
           kWindow = image((i-wSpan):(i+wSpan), (j-wSpan):(j+wSpan));
           
           k(i, j) = kernelCore(kWindow, kernel, kSize);
           
       end
       
    end

end

%
% Sobel.

function tImage = sobel(image, gSize, gDeviation)

    sobelX = [1, 0, -1; 2, 0, -2; 1, 0, -1];
	sobelY = [1, 2, 1; 0, 0, 0; -1, -2, -1];
    
    % Extend the pixel precision to double and transform to grayscale.
    
    tImage = im2double(image);
    tImage = grayscaleImage(tImage);
    
    % Gaussian blur.

    gKernel = generateGaussianKenel(gSize, gDeviation);
    tImage = applyKernel(tImage, gKernel, gSize);
    
    % Convolve with X and Y kernels.
    
    imageX = applyKernel(tImage, sobelX, 3);
    imageY = applyKernel(tImage, sobelY, 3);
    
    % Calculate the root mean square of both images and
    % diminish the precision to uint8.
    
    tImage = rootMeanSquare(imageX, imageY);
    tImage = im2uint8(tImage);

end

%
% Laplacian.

function tImage = laplacian(image, kSize, gSize, gDeviation)

    kernel = laplacianKernel(kSize);
    
    % Extend the pixel precision to double and transform to grayscale.
    
    tImage = im2double(image);
    tImage = grayscaleImage(tImage);
    
    % Gaussian blur.

    gKernel = generateGaussianKenel(gSize, gDeviation);
    tImage = applyKernel(tImage, gKernel, gSize);
    
    % Convolve and diminish the precision to uint8.
    
    tImage = applyKernel(tImage, kernel, kSize);
    tImage = im2uint8(tImage);

end

%
% Canny.

function tImage = canny_edges(image);
 
    max_hysteresis_thresh = 1.5;
    min_hysteresis_thresh = 0.05;
    sigma =1;
    
    imageSize = size(image);
    imageWidth = imageSize(2);
    imageHeight = imageSize(1);

    % Extend the pixel precision to double and transform to grayscale.
    
    tImage = im2double(image);
    tImage = grayscaleImage(tImage);            

    % Create gaussian kernels for both x and y directions based 
    % on the sigma that was given.
    
    kernelSize = 6*sigma+1;         
    y_gaussian = zeros(kernelSize, kernelSize);
    x_gaussian = zeros(kernelSize, kernelSize);
    
    for i = 1:kernelSize
        
        for j = 1:kernelSize
            
            y_gaussian(i, j) = -( (i-((kernelSize-1)/2)-1)/( 2* pi * sigma^3 ) ) * exp ( - ( (i-((kernelSize-1)/2)-1)^2 + (j-((kernelSize-1)/2)-1)^2 )/ (2*sigma^2) );
            
        end
        
    end

    for i=1:kernelSize
        
        for j=1:kernelSize
            
            x_gaussian(i, j) = -( (j-((kernelSize-1)/2)-1)/( 2* pi * sigma^3 ) ) * exp ( - ( (i-((kernelSize-1)/2)-1)^2 + (j-((kernelSize-1)/2)-1)^2 )/ (2*sigma^2) );
            
        end
        
    end

    % Derivatives in x and y directions.
    
    derivative_x = zeros(imageHeight, imageWidth);              
    derivative_y = zeros(imageHeight, imageWidth); 

    for r = 1+ceil(kernelSize/2):imageHeight-ceil(kernelSize/2)  
        
        for c = 1+ceil(kernelSize/2):imageWidth-ceil(kernelSize/2)
            
            reference_row = r-ceil(kernelSize/2); 
            reference_colum = c-ceil(kernelSize/2); 
            
            for k=1:kernelSize
                
                for k_col=1:kernelSize
                    
                    derivative_x(r, c) = derivative_x(r, c) + tImage(reference_row+k-1, reference_colum+k_col-1)*x_gaussian(k, k_col);
                    
                end
                
            end
            
        end
        
    end


    for r = 1+ceil(kernelSize/2):imageHeight-ceil(kernelSize/2) 
        
        for c = 1+ceil(kernelSize/2):imageWidth-ceil(kernelSize/2)
            
            reference_row=  r-ceil(kernelSize/2);
            reference_colum=  c-ceil(kernelSize/2);
            
            for k = 1:kernelSize  
                
                for k_col=1:kernelSize 
                    
                    derivative_y(r,c) = derivative_y(r,c) + tImage(reference_row+k-1, reference_colum+k_col-1)*y_gaussian(k,k_col);
                    
                end
                
            end
            
        end
        
    end

    % Compute the gradient magnitude based on derivatives in x and y
    % directions.
    
    gradient =  zeros(imageHeight, imageWidth);        

    for r=1+ceil(kernelSize/2):imageHeight-ceil(kernelSize/2)  
        for c=1+ceil(kernelSize/2):imageWidth-ceil(kernelSize/2)  
            gradient(r,c) = sqrt (derivative_x(r,c)^2 + derivative_y(r,c)^2 );
        end
    end

    % Perform non-maxima suppression:

    non_max = gradient;
    
    for r = 1+ceil(kernelSize/2):imageHeight-ceil(kernelSize/2)
        
        for c = 1+ceil(kernelSize/2):imageWidth-ceil(kernelSize/2)

            if (derivative_x(r,c) == 0) 
                tangent = 5;       
            else
                tangent = (derivative_y(r,c)/derivative_x(r,c));
            end
            
            if (-0.4142<tangent && tangent<=0.4142)
                if(gradient(r,c)<gradient(r,c+1) || gradient(r,c)<gradient(r,c-1))
                    non_max(r,c)=0;
                end
            end
            
            if (0.4142<tangent && tangent<=2.4142)
                if(gradient(r,c)<gradient(r-1,c+1) || gradient(r,c)<gradient(r+1,c-1))
                    non_max(r,c)=0;
                end
            end
            
            if ( abs(tangent) >2.4142)
                if(gradient(r,c)<gradient(r-1,c) || gradient(r,c)<gradient(r+1,c))
                    non_max(r,c)=0;
                end
            end
            
            if (-2.4142<tangent && tangent<= -0.4142)
                if(gradient(r,c)<gradient(r-1,c-1) || gradient(r,c)<gradient(r+1,c+1))
                    non_max(r,c)=0;
                end
            end
            
        end
        
    end

    post_hysteresis = non_max;

    for r = 1+ceil(kernelSize/2):imageHeight-ceil(kernelSize/2)  
        
        for c = 1+ceil(kernelSize/2):imageWidth-ceil(kernelSize/2) 
            
            if(post_hysteresis(r,c)>=max_hysteresis_thresh)
                
                post_hysteresis(r,c)=1;
                
            end
            
            if(post_hysteresis(r,c)<max_hysteresis_thresh && post_hysteresis(r,c)>=min_hysteresis_thresh) 
                
                post_hysteresis(r,c)=2;
                
            end
            
            if(post_hysteresis(r,c)<min_hysteresis_thresh) 
                
                post_hysteresis(r,c)=0;
                
            end
            
        end
        
    end
    
    % ...

    l = 1; 

    while (l == 1)

        l = 0;

        for r = (1+ceil(kernelSize/2)):(imageHeight-ceil(kernelSize/2))  
            
            for c = (1+ceil(kernelSize/2)):(imageWidth-ceil(kernelSize/2))  
                
                if (post_hysteresis(r, c) > 0)      
                    
                    if(post_hysteresis(r, c) == 2) 
                        
                        if( post_hysteresis(r-1, c-1) ==1 || post_hysteresis(r-1,c)==1 || post_hysteresis(r-1,c+1)==1 || post_hysteresis(r,c-1)==1 ||  post_hysteresis(r,c+1)==1 || post_hysteresis(r+1,c-1)==1 || post_hysteresis(r+1,c)==1 || post_hysteresis(r+1,c+1)==1 ) 
                            
                            post_hysteresis(r,c)=1;
                            
                            l = 1;
                            
                        end
                        
                    end
                    
                end
                
            end
            
        end

    end

    tImage = post_hysteresis;
    tImage = im2uint8(tImage);

end





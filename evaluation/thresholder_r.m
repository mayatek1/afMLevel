%  DESCRIPTION:   Function written for NanoLocz: Localization Atomic Force Microscopy Analysis Platform
%  AUTHOR:        George Heath, University of Leeds,   g.r.heath@leeds.ac.uk,   30.06.2023

% 'Edge' method code is adapted and based on the FindSteps.m script from SPIW
% (https://sourceforge.net/projects/spiw/)
% Copyright (C) Richard Woolley & Julian Stirling
% SPIW is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.%

function [imgt] = thresholder_r(img, method, limits, invert)
 imgt =img;

persistent py_module

switch method
    case 'selection'
        imgt = imgt.*1;       imgt(imgt==0)= nan;
    case 'histogram'
        min_max = [limits(1) limits(2)];
        imgt = (img<=min_max(2)).*(img>=min_max(1));
        imgt(imgt==0) = NaN;

    case 'otsu'
        min_max(2) = multithresh(img,1);
        min_max(1) = -inf;
        imgt = (img<=min_max(2)).*(img>=min_max(1));
        imgt(imgt==0) = NaN;

    case 'auto edges'
        h= imgaussfilt(img,2);
        sob = fspecial('sobel');
        xgrad = imfilter(h,sob','replicate');
        ygrad = imfilter(h,sob,'replicate');
        ygrad = ygrad -0.5*median(ygrad,2);
        Grad = (2*xgrad.^2)+ygrad.^2;
        IM = Grad;
        thresh = min(IM(:))+(mean(IM(:))-min(IM(:)))*1.5;
        BW = IM>thresh;
        BW = bwareaopen(BW,100);
        BW = ~bwareaopen(~BW,50);
        se = strel('disk', 5);
        BW = imclose(BW, se);
        BW = imdilate(BW,se);
        BW = imclose(BW, se);
         BW =bwmorph(BW,'bridge');
        imgt = double(~BW);
        imgt(imgt==0) = NaN;

    case 'otsu edges'
        h = imgaussfilt(img,2);
        min_max(2) = multithresh(h,1);
        min_max(1) = -inf;
        imgt = (h<=min_max(2)).*(h>=min_max(1));
        imgt = ~imgt;

        if numel(size(img)) ==2
            BW = bwmorph(imgt,'remove');
            BW = bwareaopen(BW,100);
            BW = ~bwareaopen(~BW,50);
            se = strel('disk', 3);
            BW = imdilate(BW,se);
                    BW = bwareaopen(BW,100);
        BW = ~bwareaopen(~BW,50);
        else

            BW = imgt;
            for i = 1:size(img,3)
                BW(:,:,i) = bwmorph(imgt(:,:,i),'remove');
                BW(:,:,i) = bwareaopen(BW(:,:,i),100);
                BW(:,:,i) = ~bwareaopen(~BW(:,:,i),50);
                se = strel('disk', 3);
                BW(:,:,i) = imdilate(BW(:,:,i),se);
            end
        end
        imgt = double(~BW);
        imgt(imgt==0) = NaN;

    case 'ML'
        if isempty(py_module)
            py_module = py.importlib.import_module('afmlevel_functions'); % changed from ApplyLevellingModels
        end
        py_img = py.numpy.array(img, py.numpy.float32);
        py_imgt = py_module.applymodel_mask(py_img, 'mask_unet'); % run python mask function
        imgt_uint8 = uint8(py.numpy.asarray(py_imgt, py.numpy.uint8));
        imgt_double = double(logical(imgt_uint8));
        imgt = imgt_double;
        imgt(imgt==0) = NaN;

    case 'ML edges'
        if isempty(py_module)
            py_module = py.importlib.import_module('afmlevel_functions');
        end
        py_img = py.numpy.array(img, py.numpy.float32);
        py_imgt = py_module.applymodel_mask(py_img, 'mask_unet');
        imgt_uint8 = uint8(py.numpy.asarray(py_imgt, py.numpy.uint8));
        imgt_double = double(logical(imgt_uint8));
        imgt = imresize(imgt_double, size(img), 'nearest');
        imgt = ~imgt;

        if numel(size(img)) ==2
            BW = bwmorph(imgt,'remove');
            BW = bwareaopen(BW,100);
            BW = ~bwareaopen(~BW,50);
            se = strel('disk', 3);
            BW = imdilate(BW,se);
                    BW = bwareaopen(BW,100);
        BW = ~bwareaopen(~BW,50);
        else

            BW = imgt;
            for i = 1:size(img,3)
                BW(:,:,i) = bwmorph(imgt(:,:,i),'remove');
                BW(:,:,i) = bwareaopen(BW(:,:,i),100);
                BW(:,:,i) = ~bwareaopen(~BW(:,:,i),50);
                se = strel('disk', 3);
                BW(:,:,i) = imdilate(BW(:,:,i),se);
            end
        end
        imgt = double(~BW);
        imgt(imgt==0) = NaN;

    case 'MLbg'
        py_img = py.numpy.array(img, py.numpy.float32);
        py_module = py.importlib.import_module('afmlevel_functions'); % import the python script
        py_bg = py_module.applymodel_bg(py_img, 'background_unet'); % run python bg function
        bg = double(py.numpy.asarray(py_bg, py.numpy.float64));
        imgt = bg;
        % instead of a mask, the output here is the background


    case 'stddev'
        me = mean(img(:));
        sd = std(double(img(:)));


        if limits(1) == -inf,
            min_val = -inf;
        else
            min_val = me - limits(1) * sd;
        end

        max_val = me + limits(2) * sd;

        imgt = (img >= min_val).*(img <= max_val);
        imgt(imgt == 0) = NaN;

end

if invert ==1
    pos = isnan(imgt);
    imgt =pos.*img;
    imgt(imgt==0) = NaN;
end


end
function [result, imgt] = level_auto_r(img,filter_frames,routine)
imgt = [];
%choose imgt_all if needed

%just leave blank and use function as normal if not using ML mask
result = img;
switch routine

 % No mask
    case 'plane-line'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            prev = level(prev,0,0,'med_line');
            result(:,:,i) = prev;
            imgt = 0;
        end

 % Threshold mask
     case 'iterative stdev'

        limits = [-inf,2];

        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');

            imgt = thresholder_r(prev,'stddev',limits,0);
            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'stddev',limits,0);
            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'stddev',limits,0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,1,0,'plane',imgt);

            imgt = thresholder_r(prev,'stddev',limits,0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,2,0,'plane',imgt);
            result(:,:,i) = prev;
        end


    case {'iterative fixed high','Iterative fixed low','Iterative fixed high low'}
       switch lower(routine)
            case 'iterative fixed high'
                low = -Inf;
                high = 0.75;
            case 'iterative fixed low'
                low = -0.75;
                high = Inf;
            case 'iterative fixed high low'
                low = -0.75;
                high = 0.75;
        end
        for i = filter_frames

            [h, w] = size(img);

            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');

            imgt = thresholder_r(prev,'histogram',[low, high],0);
            prev = level(prev,1,1,'plane',imgt);
            imgt = thresholder_r(prev,'histogram',[low, high],0);

            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'histogram',[low, high],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,1,0,'plane',imgt);

            imgt = thresholder_r(prev,'histogram',[low, high],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,2,0,'plane',imgt);
            result(:,:,i) = prev;
        end


         case 'gaussian high low'
        try
            for i = filter_frames
                prev = img(:,:,i);
                prev= level(prev,1,1,'plane');
                prev= level(prev,0,0,'med_line');
                result(:,:,i) = prev;
            end
            fo = fitoptions('method','NonlinearLeastSquares');
            t = result(:,:,:);
            [hy, x]=  hist(double(t(:)),100);
            gfit  = fit(double(x'),hy','gauss1',fo);
            max = gfit.b1 +1.5*gfit.c1;
            min = gfit.b1 -1.5*gfit.c1;

            for i = filter_frames
                prev = result(:,:,i);
                imgt = thresholder_r(prev,'histogram',[min, max],0);
                prev = level(prev ,1,1,'plane',imgt);
                prev = level(prev ,0,0,'med_line',imgt);
                result(:,:,i) = prev;
            end
        catch e
              warning('Error occurred: %s', e.message);
              result(:, :, i) = 0;
        end


    case {'iterative gaussian low', 'iterative gaussian high'}
        switch routine
            case 'iterative gaussian low'
                h = 0;
                p = Inf;
            case 'iterative gaussian high'
                h = -Inf;
                p = 0;
        end
        try
            for i = filter_frames
                prev = img(:,:,i);
                prev= level(prev,2,2,'plane');
                prev= level(prev,0,0,'med_line');
                result(:,:,i) = prev;
            end
            fo = fitoptions('method','NonlinearLeastSquares');
            t = result(:,:,:);
            [hy, x]=  hist(double(t(:)),100);
            gfit  = fit(double(x'),hy','gauss1',fo);
            max = (gfit.b1 +1.5*gfit.c1)+p;
            min = (gfit.b1 -1.5*gfit.c1)+h;

            for i = filter_frames
                prev = result(:,:,i);
                imgt = thresholder_r(prev,'histogram',[min, max],0);
                prev = level(prev ,2,2,'plane',imgt);
                prev = level(prev ,0,0,'med_line',imgt);
                result(:,:,i) = prev;
            end
            t = result(:,:,:);
            [hy, x]=  hist(double(t(:)),100);
            gfit  = fit(double(x'),hy','gauss1',fo);
            max = (gfit.b1 +1.5*gfit.c1)+p;
            min = (gfit.b1 -1.5*gfit.c1)+h;

            for i = filter_frames
                prev = result(:,:,i);
                imgt = thresholder_r(prev,'histogram',[min, max],0);
                prev = level(prev ,2,2,'plane',imgt);
                prev = level(prev,1,0,'line',imgt);
                result(:,:,i) = prev;
            end
        catch e
              warning('Error occurred: %s', e.message);
              result(:, :, i) = 0;
        end


        case 'iterative otsu'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');

            imgt = thresholder_r(prev,'otsu',[],0);
            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'otsu',[],0);
            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'otsu',[],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,1,0,'plane',imgt);

            imgt = thresholder_r(prev,'otsu',[],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,2,0,'plane',imgt);
            result(:,:,i) = prev;
        end


    case 'otsu + line'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,0,'line');
            imgt = thresholder_r(prev,'otsu',[],0);
            prev = level(prev,2,0,'line',imgt);
            result(:,:,i) = prev;
        end




  %Multi-plane
    case 'multi-plane edges'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            imgt = thresholder_r(prev,'auto edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'auto edges',['-inf', 'inf'],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            result(:,:,i) = prev;
        end

    case 'multi-plane edges + line'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            prev = level(prev,0,0,'med_line');
            imgt = thresholder_r(prev,'auto edges',[0 0],0);
            imgt1 = imgt;
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'auto edges',['-inf', 'inf'],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            result(:,:,i) = prev;
        end


    case 'multi-plane otsu'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            result(:,:,i) = prev;
        end

    case 'multi-plane otsu + line'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            prev = level(prev,0,0,'med_line');
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            imgt = thresholder_r(prev,'otsu edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            result(:,:,i) = prev;
        end


 %MLMask
    case 'plane-line MLMask'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');

            imgt = thresholder_r(prev,'ML',[],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,1,1,'plane', imgt);

            result(:,:,i) = prev;
        end

    case 'iterative MLMask'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');

            imgt = thresholder_r(prev,'ML',[],0);
            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'ML',[],0);
            prev = level(prev,1,1,'plane',imgt);

            imgt = thresholder_r(prev,'ML',[],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,1,0,'plane',imgt);

            imgt = thresholder_r(prev,'ML',[],0);
            prev = level(prev ,0,0,'med_line',imgt);
            prev = level(prev,2,0,'plane',imgt);
            result(:,:,i) = prev;
        end


    case 'multi-plane MLMask'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            result(:,:,i) = prev;
        end


    case 'multi-plane MLMask + line'
        for i = filter_frames
            prev = img(:,:,i);
            prev = level(prev,1,1,'plane');
            prev = level(prev,0,0,'med_line');
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            imgt = thresholder_r(prev,'ML edges',[0 0],0);
            prev = level_weighted(prev,2,2,'plane',imgt);
            prev = level_weighted(prev,0,0,'med_line',imgt);
            result(:,:,i) = prev;
        end

%MLBackground
    case 'MLBackground 1x'
        for i = filter_frames
            prev = img(:,:,i);
            imgt = thresholder_r(prev,'MLbg',[],0);
            prev = level(prev,1,1,'plane');
            prev = prev-imgt;
            result(:, :, i) = prev - mean(prev(:)); %set mean to 0


        end

    case 'MLBackground 2x'
        for i = filter_frames
            prev = img(:,:,i);
            imgt = thresholder_r(prev,'MLbg',[],0);

            prev = level(prev,1,1,'plane');
            prev = prev-imgt;
            imgt = thresholder_r(prev,'MLbg',[],0);

            prev = level(prev,1,1,'plane');
            prev = prev-imgt;
            result(:, :, i) = prev - mean(prev(:)); %set mean to 0
        end

    case 'MLBackground 3x'
        for i = filter_frames
            prev = img(:,:,i);
            imgt = thresholder_r(prev,'MLbg',[],0);
            prev = level(prev,1,1,'plane');
            prev = prev-imgt;
            imgt = thresholder_r(prev,'MLbg',[],0);
            prev = level(prev,1,1,'plane');
            prev = prev-imgt;
            imgt = thresholder_r(prev,'MLbg',[],0);
            prev = level(prev,1,1,'plane');
            prev = prev-imgt;
            result(:, :, i) = prev - mean(prev(:)); %set mean to 0
        end

end
end
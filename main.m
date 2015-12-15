clear;
% Meyer, Nadro, Kuck 2015
% CS445 Computational Photography
% Mainfile for grabcut
addpath(genpath('GCMex'));

DO_GRABCUT  = true;

img = imread('tree.jpg');

%Run grab cut.
if DO_GRABCUT
    figure(1), imshow(img);
    h = imrect;
    box = logical(createMask(h));
    pos = getPosition(h);
    figure(1), imshow(box);
    result = grabcut(img,box,pos);
    figure(1), imagesc(result);
    combined = img .* repmat(result,1,1,3);
    figure(2), imshow(combined);
    
    imwrite(combined,'final_result.tiff');
end


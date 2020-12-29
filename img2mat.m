rand("seed", 1);
img = imread("Lenna.png");
for i = 1: 3
    writetable(table(img(:, :, i)), "image\img" + num2str(i) + ".csv");
end
img1 = double(img) + randn(size(img)) * 0.1;
for i = 1: 3
    writetable(table(img1(:, :, i)), "image\img_noise" + num2str(i) + ".csv");
end
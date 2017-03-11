
function region = enlargingbbox(bbox, scale)

region(1) = bbox(1) - (scale-1)/2*bbox(3);
region(2) = bbox(2) - (scale-1)/2*bbox(4);

region(3) = scale*bbox(3);
region(4) = scale*bbox(4);

end
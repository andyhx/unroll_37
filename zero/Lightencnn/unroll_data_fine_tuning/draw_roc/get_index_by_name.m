
function index=get_index_by_name(name)

img_name=name;
if(strcmp(img_name(1:3),'BRL'))
    index=str2num(img_name(5:8));
elseif strcmp(img_name(1:4),'BU3D')
    index=str2num(img_name(6:9))+124;
elseif strcmp(img_name(1:4),'BU4D')
    index=str2num(img_name(6:9))+124+100;
elseif strcmp(img_name(1:4),'MICC')
    index=str2num(img_name(6:9))+124+100+101;
end

end

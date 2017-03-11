function result=write_index(img)

for i_p=1:length(img)
   img(i_p).index=get_index_by_name(img(i_p).name);
end
result=img;
end

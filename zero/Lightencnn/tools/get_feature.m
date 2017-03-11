function result=get_feature(basic_dir,cnnModel,weights)

feature=dir([basic_dir filesep '*.jpg']);
feature=write_index(feature);
result=extract_feature(basic_dir,feature,cnnModel,weights);

end

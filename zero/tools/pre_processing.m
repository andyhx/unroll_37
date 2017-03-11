face_dir = '/home/brl/data/LFW/having_pts';
ffp_dir = '/home/brl/data/LFW/having_pts';
ec_mc_y = 48; %%��������(ec)�������ģ�mc��֮��ľ���
ec_y = 40; %%�������ĵ�Y���
img_size = 128; %%light CNN������ͼ��Ϊ128*128
save_dir = '/home/brl/data/LFW/temp'; %%������ͼ��ı���·��

res = face_db_align(face_dir, ffp_dir, ec_mc_y, ec_y, img_size, save_dir)
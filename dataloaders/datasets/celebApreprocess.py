import random

def parse_name_list(fp):
    with open(fp, 'r') as fin:
        lines = fin.readlines()
    parsed = list()
    for line in lines:
        parsed.append(line.strip().split(".")[0])
    return parsed

def write_dir(dir, name_list):
	with open(dir, 'w+') as fout:
		for name in name_list:
			fout.write("%s\n" % name)
		fout.close()

def split(txt_dir, val_dir, test_dir, train_dir):
	name_list = parse_name_list(txt_dir)
	random.shuffle(name_list)
	write_dir(val_dir, name_list[:500])
	write_dir(test_dir, name_list[500:1000])
	write_dir(train_dir, name_list[1000:])
	

split("/Users/yulian/Desktop/Courses/CS231N/231n_lucky_project/data/celebA/segmentations.txt", 
	  "/Users/yulian/Desktop/Courses/CS231N/231n_lucky_project/data/celebA/val.txt",
	  "/Users/yulian/Desktop/Courses/CS231N/231n_lucky_project/data/celebA/test.txt",
	  "/Users/yulian/Desktop/Courses/CS231N/231n_lucky_project/data/celebA/train.txt")


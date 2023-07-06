import sys


IN_COLAB = 'google.colab' in sys.modules

if not IN_COLAB:

    from git import Repo

    # Initialize the Git repository object
    repo = Repo(".", search_parent_directories=True)

    # Get the root directory of the Git project
    root_dir = repo.git.rev_parse("--show-toplevel")

    from pathlib import Path

    # Set up path for custom importer modules
    importer_module = root_dir + '/dataloader/'
    sys.path.insert(0, importer_module)

    # Insert here your local path to the dataset
    data_path = '/home/leeoos/Desktop/'

else: 
    
    from google.colab import drive
    drive.mount('/content/drive')

    # On Colab the path to the module si fixed once you have 
    # correttly set up the project with gitsetup.ipynb 
    fixed_path = '/content/drive/MyDrive/Github/visiope/dataloader/'
    sys.path.insert(0, fixed_path)

    # Insert here the path to the dataset on your drive
    data_path = '/content/drive/MyDrive/Dataset/'

from asloader import Ai4MarsImporter, Ai4MarsProcessor, Ai4MarsData


data_import = Ai4MarsImporter()

X, y = data_import(path=data_path, num_of_images=200)


processor = Ai4MarsProcessor(X, y)

train_set, test_set, val_set = processor([0.54, 0.26, 0.20])

# print(len(train_set))
# print(type(train_set))

# from torch.utils.data import DataLoader


# train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)




# IMPORT

# percentages = [0.54, 0.26, 0.20]

# # assertions
# assert math.ceil(sum(percentages)) == 1.

# # Paths
# images = "ai4mars-dataset-merged-0.1/msl/images"
# label_train = "ai4mars-dataset-merged-0.1/msl/labels/train"
# label_test_1ag = "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min1-100agree"
# label_test_2ag = "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min2-100agree"
# label_test_3ag = "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree"
# edr = images + "/edr"
# mxy = images + "/mxy"
# rng = images + "/rng-30m"

# # In this way we collect list of al files in the projects
# edr_files = os.listdir(edr)
# label_train_files = os.listdir(label_train)
# label_test_files_1 = os.listdir(label_test_1ag)
# label_test_files_2 = os.listdir(label_test_2ag)
# label_test_files_3 = os.listdir(label_test_3ag)

# assert len(label_train_files) % 2 == 0

# # If we want to split before two create separate datasets
# if percentages:
#     subsets_X = [[] for i in range(len(percentages))]
#     subsets_y = [[] for i in range(len(percentages))]

# else:
#     X = []  # input
#     y = []  # label train

# # This cannot be shuffled
# y1 = [] # test label 1 agree
# y2 = [] # test label 2 agree
# y3 = [] # test label 3 agree

# # This count how many images insert
# image_counter = 0 

# for label1, label2, label3 in zip(label_test_files_1, label_test_files_2, label_test_files_3):
#     path_label1 = os.path.join(label_test_1ag, label1)
#     img_arr = cv2.imread(path_label1, 0) 
#     y1.append(img_arr)

#     path_label2 = os.path.join(label_test_2ag, label2)
#     img_arr = cv2.imread(path_label2, 0) 
#     y2.append(img_arr)

#     path_label3 = os.path.join(label_test_3ag, label3)
#     img_arr = cv2.imread(path_label3, 0)
#     y3.append(img_arr)

# for label in label_train_files:
#     # Names of images match names of labels, except for the extension (JPG, png)
#     img_name = label[:-4] + ".JPG" 

#     if img_name in edr_files:
#         img_path = os.path.join(edr, img_name) 
#         img_arr = cv2.imread(img_path) 

#         label_path = os.path.join(label_train, label)
#         lab_arr = cv2.imread(label_path, 0) # 0 mean read as greyscale image

#         # Build inputs, labels tensors eighter splitted or not
#         if percentages:
#             random_index = random.uniform(0,1)
#             start = 0
            
#             for i in range(len(percentages)):
#                 if start <= random_index <= percentages[i] + start:
#                     random_index = i
#                     break
#                 start += percentages[i]

#             subsets_X[random_index].append(torch.from_numpy(img_arr))
#             subsets_y[random_index].append(torch.from_numpy(lab_arr[:, :, np.newaxis]))

#         else:
#             X.append(torch.from_numpy(img_arr))
#             y.append(torch.from_numpy(lab_arr[:, :, np.newaxis]))
       

#         # Check if images and labes corresponds
#         # if image_counter == 3:
#         #     print(img_arr.shape)
#         #     print(lab_arr.shape)
#         #     print(X[3].shape)
#         #     print(y[3].shape)
#         #     plt.imshow(img_arr, cmap='gray')
#         #     plt.show()
#         #     plt.imshow(lab_arr, cmap='gray')
#         #     plt.show()
#         #     plt.imshow(torch.from_numpy(img_arr), cmap='gray')
#         #     plt.show()
#         #     plt.imshow(torch.from_numpy(lab_arr), cmap='gray')
#         #     plt.imshow(lab_arr, cmap='gray')
#         #     plt.show()
       
#         image_counter += 1  # this control how much images you want
#         if image_counter == 200: break

    
# # Normalization and assigment of right label to background
# if percentages:

#     for i in range(len(percentages)):
#         subsets_X[i] = torch.stack(subsets_X[i], dim=0)
#         subsets_X[i] = subsets_X[i] / 255

#         subsets_y[i] = torch.stack(subsets_y[i], dim=0)
#         subsets_y[i][subsets_y[i] == 255] = 4

# else:
#     X = torch.stack(X, dim=0)
#     X = X / 255

#     y = torch.stack(y, dim=0)
#     y[y == 255] = 4


# # test
# if percentages:
    
#     for i in range(len(percentages)):
#         print(subsets_X[i].shape)
#         print(subsets_y[i].shape)
# else:
#     print(X.shape)
#     print(y.shape)
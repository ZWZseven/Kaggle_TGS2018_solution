from tqdm import tqdm_notebook


train_df11 = pd.read_csv("../input/tgs-salt-identification-challenge/train.csv", index_col="id", usecols=[0])
depths_df11 = pd.read_csv("../input/tgs-salt-identification-challenge/depths.csv", index_col="id")
#train_df
train_df11 = train_df11.join(depths_df11)
test_df11 = depths_df11[~depths_df11.index.isin(train_df11.index)]

df_pred = pd.read_csv('../input/tgs-salt-identification-challenge/train.csv', index_col=[0])
df_pred.fillna('', inplace=True)
df_pred['suspicious'] = False

i=0
for index, row in df_pred.iterrows():
    encoded_mask = row['rle_mask'].split(' ')
    i=i+1
    mask0 = rle_decode(row['rle_mask'])

    if (len(encoded_mask) > 1 and len(encoded_mask) < 5 and int(encoded_mask[1]) % 101 == 0 and int(encoded_mask[1]) <= 100*101):
        df_pred.loc[index,'suspicious'] = True
        
img_size_ori = 101
img_size_target = 128#127

train_df0 = pd.read_csv("../input/tgs-salt-identification-challenge/train.csv", index_col="id", usecols=[0])
depths_df0 = pd.read_csv("../input/tgs-salt-identification-challenge/depths.csv", index_col="id")
#train_df
train_df0 = train_df0.join(depths_df0)
test_df0 = depths_df0[~depths_df0.index.isin(train_df0.index)]
train_df0["images"] = [np.array(cv2.imread("../input/tgs-salt-identification-challenge/train/images/{}.png".format(idx), 0)) / 255 for idx in (train_df0.index)]
#print(train_df.shape) if ~df_pred.loc[idx,'suspicious']
train_df0['suspicious'] = df_pred['suspicious']
train_df0['rle_mask'] = df_pred['rle_mask']
train_df0["masks"] = [np.array(cv2.imread("../input/tgs-salt-identification-challenge/train/masks/{}.png".format(idx), 0)) / 255 for idx in (train_df0.index)]

train_df0["coverage"] = train_df0.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df0["coverage_class"] = train_df0.coverage.map(cov_to_class)
##############################################################################
train_df1=train_df0.copy()
sus_df = train_df0[train_df0.suspicious]  

################################################################

imtrain=np.array(train_df0.images.tolist())########################################
immask=np.array(train_df0.masks.tolist())
#np.array(train_df.masks.map(imgexpand).tolist()).reshape(-1, img_size_target, img_size_target, 1)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df0.index.values,
    imtrain, 
    immask, 
    train_df0.coverage.values,
    train_df0.z.values,
    test_size=0.08, stratify=train_df0.coverage_class, random_state=1337)


#########################################################
train_df0.reset_index(inplace=True)

ids_train=pd.DataFrame(ids_train)
#ids_train.rename(columns={'0':'id'},inplace=True)
ids_train.columns = ['id']#

ids_valid=pd.DataFrame(ids_valid)
#ids_train.rename(columns={'0':'id'},inplace=True)
ids_valid.columns = ['id']#

train_df=train_df0[train_df0.id.isin(ids_train.id)]
#df.drop(['B','C'],axis=1,inplace=True)
valid_df=train_df0[train_df0.id.isin(ids_valid.id)]

train_df1.reset_index(inplace=True)
valid_df_sus=train_df1[~train_df1.id.isin(ids_train.id)]
##################################################################
grp = list(train_df.groupby('id'))

zmin=min([list(m['z'].values) for _,m in grp])
zmax=max([list(m['z'].values) for _,m in grp])
zdif=(zmax[0]-zmin[0])
print(zdif)
print(zmin)

#################################################################

train_loader = make_loader(train_df, batch_size =  BATCH_SIZE, shuffle=True, transform=train_transform)
valid_loader = make_loader(valid_df, batch_size = BATCH_SIZE // 2, transform=None)

########################################################
train_transform = DualCompose([
        HorizontalFlip(),
        ZoominRandomCropNew([101,101],1.10),
        Rotate(limit=10),
        RandomBrightness(limit=0.08),
])

val_transform = DualCompose([
        #Imgexpand(),##############################################CenterCrop((512,512,3)),
      ])


######################################################
model = LinkNet34deeps(1) #


###################################################
model_path ='../input/linknet34deepsv3fulldare60002d01scalere4/model_1.pt'#'../input/model0924/model_0924denseHlight.pt'
state = torch.load(str(model_path))
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
if torch.cuda.is_available():
    model.cuda()


retrain5(init_optimizer=lambda lr: torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, momentum=0.9, weight_decay=0.0001),#(model.parameters(), lr=lr),
        lr = 1e-2,#1e-4
        n_epochs = 60,#35,#40,
        model=model,
        criterion=criterionL5,#LossBinaryElu(jaccard_weight=0),
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation5,
     )

####log visualization####################################
log_file = 'train_1.log'
logs = pd.read_json(log_file, lines=True)

plt.figure(figsize=(26,6))
plt.subplot(1, 2, 1)
plt.plot(logs.step[logs.loss.notnull()],
            logs.loss[logs.loss.notnull()],
            label="on training set")
 
#plt.plot(logs.step[logs.valid_loss.notnull()],
#            logs.valid_loss[logs.valid_loss.notnull()],
#            label = "on validation set")
         
plt.xlabel('step')
plt.legend(loc='center left')
plt.tight_layout()
plt.show();

###################################################################################

######################################################################

model = LinkNet34deeps(1)#UnetPP(1)#DenseNet34() #LinkNet34a(1)#Unet_grid_attention()#DenseNet34H(1)#Incv3( num_classes=1, num_channels=3)
#model = UNet34H(1)#DenseNet34H(1)##LinkNet34(1)#AlbuNet()#UNetResNet(34,1,dropout_2d=0.0)#UNet()
model_path ='model_1.pt'
state = torch.load(str(model_path))
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
if torch.cuda.is_available():
    model.cuda()

model.eval()

train_df1=train_df1.set_index('id')


from skimage.morphology import binary_opening, disk

loader = DataLoader(
        dataset=SaltDataset(valid_df, transform=None, mode='valid'),###########################################################
        shuffle=False,
        batch_size=16,#BATCH_SIZE,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    ) 

y_valid_ori = []
#preds_valid = [] #
out_pred_rows = []#pd.DataFrame(columns=('id', 'mask'))#[]
for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='valid')):
    inputs = variable(inputs, volatile=True)
    outputs = dummy_prediction2(inputs,model)

    for i, image_name in enumerate(paths):
        mask = outputs[i,0]#F.sigmoid(outputs[i,0]).data.cpu().numpy()
        out_pred_rows.extend([restore(mask).tolist()])# += [{'id': image_name, 'mask': mask}]
        y_valid_ori.extend([train_df1.loc[image_name].masks])


y_valid_ori = np.array(y_valid_ori)
preds_valid = np.array(out_pred_rows)

#######################################################
thresholds = np.linspace(0, 1, 50)
ious = np.array([get_iou_vector(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

##test dataset########################################################
test_paths = os.listdir(test_image_dir+'/images/')
print(len(test_paths), 'test images found')

test_df0['rle_mask'] = None

train_df=0
train_df0=0
train_df1=0
grp=0

if not('id' in test_df0.columns.values.tolist()):
    test_df0.reset_index(inplace=True)

from skimage.morphology import binary_opening, disk
#
loader = DataLoader(
        dataset=SaltDataset(test_df0, transform=None, mode='predict'),
        shuffle=False,
        batch_size=16,#BATCH_SIZE,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    ) 
preds_test = {}#[] 
#ind = []
out_pred_rows = []
for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
    inputs = variable(inputs, volatile=True)
    outputs = dummy_prediction2(inputs,model)
    #outputs = model(inputs)
    for i, image_name in enumerate(paths):

        mask = outputs[i,0]#
        if image_name[-4:]=='.png':
            preds_test[image_name[:-4]]=(restore(mask))
        else:
            preds_test[image_name]=(restore(mask))

co=0
for i,idx in enumerate(test_df11.index.values):
    num=np.sum(np.sum(np.round(preds_test[idx] > threshold_best)))
    #print(num)
    if num<20 and num>0:
        co=co+1
        preds_test[idx] =np.zeros((101,101))
print(co)


pred_dict = {idx: RLenc(np.round(preds_test[idx] > threshold_best)) for i, idx in enumerate(tqdm(test_df11.index.values))}#.id))}


sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submissionpth.csv')



    


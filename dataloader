def imgexpand03depth(img,depth):
    t_size=128#224
    output0=np.zeros([t_size,t_size,3])#([192,192,3])#([127,127])
    output=imgexpand0(img.squeeze()).squeeze()#np.zeros([128,128])#
    output0[:,:,0]=output
    output0[:,:,1]=output*depth
    output0[:,:,2]=output*depth
    return output0

def imgexpand03d(img):
    t_size=128#224
    output0=np.zeros([t_size,t_size,3])#([192,192,3])#([127,127])
    output=imgexpand0(img.squeeze()).squeeze()#np.zeros([128,128])#
    output0[:,:,0]=output
    output0[:,:,1]=output
    output0[:,:,2]=output
    return output0

def add_depth_channels(image_tensor):
    h, w,_ = image_tensor.shape#size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[row, :,1] = const
    image_tensor[:,:,2] = image_tensor[:,:,0] * image_tensor[:,:,1]
    return image_tensor

def C_add_depth_channels(image_tensor,depth):
    h, w,_ = image_tensor.shape#size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[row, :,1] = const
    image_tensor[:,:,2] = image_tensor[:,:,0] * image_tensor[:,:,1]
    image_tensor[:,:,1] = image_tensor[:,:,1] * depth
    return image_tensor

class SaltDataset(Dataset):
    def __init__(self, in_df, transform=None, mode='train', zdif=908, zmin=51):
        grp = list(in_df.groupby('id'))
        
        self.image_ids =  [_id for _id, _ in grp] 
        self.z = [m['z'].values for _,m in grp]
        self.zmin = zmin
        self.zdif = zdif
        self.image_masks = [m['rle_mask'].values for _,m in grp]
        self.transform = transform
        self.mode = mode
        self.img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)
               
    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if self.mode == 'train':
            rgb_path = os.path.join(train_image_dir, 'images',img_file_name+'.png')
        elif self.mode == 'valid':
            rgb_path = os.path.join(train_image_dir, 'images',img_file_name+'.png')
        else:
            if img_file_name[-4:]=='.png':
                rgb_path = os.path.join(test_image_dir, 'images',img_file_name)
            else:
                rgb_path = os.path.join(test_image_dir, 'images',img_file_name+'.png')
        img = imread(rgb_path)
        mask = masks_as_image(self.image_masks[idx])  
        depth = self.z[idx]
        depth = math.ceil((depth-self.zmin)/self.zdif*101.)/101.
        if depth==0:
            depth=1./101.
        #print(depth)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        #img = imgexpand03(img[:,:,0])###################################################################
        #img = imgexpand03depth(img[:,:,0],depth)#
        img = imgexpand03d(img[:,:,0])
        img = C_add_depth_channels(img,depth)#
        size0=192
        img = cv2.resize(img[:, :, :], (size0, size0), interpolation=cv2.INTER_LINEAR)
        
        mask =imgexpand0(mask)#imgexpand256(mask)###################################################################
        mask = cv2.resize(mask, (size0, size0), interpolation=cv2.INTER_LINEAR)
        if mask is not None:           
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)            
        if img is not None:           
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)                   
                
        if self.mode == 'train':
            #return self.to_float_tensor(img), self.to_float_tensor(mask)
            #eturn img, mask
            #print(mask.shape)
            return self.img_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()
        else:
            return self.img_transform(img), str(img_file_name)




def make_loader(in_df, batch_size, shuffle=False, transform=None):
        return DataLoader(
            dataset=SaltDataset(in_df, transform=transform),
            shuffle=shuffle,
            num_workers = 0,
            batch_size = batch_size,
            pin_memory=torch.cuda.is_available()
        )



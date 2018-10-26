import argparse
import time
import datetime

from data_preparation.dataloader import *
from data_preparation.data_augmentation import *
from model.unet34_deepsupervision import *
from model.model import *
from eval import *
from loss import *
from prediction import *
from train import *
from utils import *

from tqdm import tqdm_notebook
from skimage.morphology import binary_opening, disk

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
class ModelSolver(object):
    def __init__(self, config):
        self.traincsv_path = config.traincsv_path
        self.depthscsv_path = config.depthscsv_path
        self.train_image_path = config.train_image_path
        self.train_mask_path = config.train_mask_path
        self.test_image_dir = config.test_image_dir
        
        self.model_path = config.pretrained_model
        
        # Model hyper-parameters
        self.image_size = config.image_size

        # Hyper-parameteres
        self.g_lr = config.lr
        #self.cycle_num = config.cycle_num
        #self.cycle_inter = config.cycle_inter
        self.epochs = config.epochs

        self.batch_size = config.batch_size


    def data_prep(self, traincsv_path, depthscsv_path, train_image_path, train_mask_path):
        train_df11 = pd.read_csv(traincsv_path, index_col="id", usecols=[0])
        depths_df11 = pd.read_csv(depthscsv_path, index_col="id")
        #train_df
        train_df11 = train_df11.join(depths_df11)
        test_df11 = depths_df11[~depths_df11.index.isin(train_df11.index)]

        df_pred = pd.read_csv(traincsv_path, index_col=[0])
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

        train_df0 = pd.read_csv(traincsv_path, index_col="id", usecols=[0])
        depths_df0 = pd.read_csv(depthscsv_path, index_col="id")
        #train_df
        train_df0 = train_df0.join(depths_df0)
        test_df0 = depths_df0[~depths_df0.index.isin(train_df0.index)]
        train_df0["images"] = [np.array(cv2.imread(train_image_path+"/{}.png".format(idx), 0)) / 255 for idx in (train_df0.index)]
        #print(train_df.shape) if ~df_pred.loc[idx,'suspicious']
        train_df0['suspicious'] = df_pred['suspicious']
        train_df0['rle_mask'] = df_pred['rle_mask']
        train_df0["masks"] = [np.array(cv2.imread(train_mask_path+"/{}.png".format(idx), 0)) / 255 for idx in (train_df0.index)]

        train_df0["coverage"] = train_df0.masks.map(np.sum) / pow(img_size_ori, 2)

        train_df0["coverage_class"] = train_df0.coverage.map(cov_to_class)
        ##############################################################################
        train_df1=train_df0.copy()
        sus_df = train_df0[train_df0.suspicious]  

        ##################################################################
        n_fold = 5
        depths = pd.read_csv(depthscsv_path)
        depths.sort_values('z', inplace=True)
        #depths.drop('z', axis=1, inplace=True)
        depths['fold'] = (list(range(n_fold))*depths.shape[0])[:depths.shape[0]]

        print(depths.head())

        train_df = depths[depths.fold!=0]
        train_df = train_df[train_df.id.isin(train_df0.index)]
        train_df= train_df.join(df_pred,on='id')

        valid_df = depths[depths.fold==0]
        valid_df = valid_df[valid_df.id.isin(train_df0.index)]
        valid_df= valid_df.join(df_pred,on='id')

        train_df0.reset_index(inplace=True)

        grp = list(train_df.groupby('id'))

        zmin=min([list(m['z'].values) for _,m in grp])
        zmax=max([list(m['z'].values) for _,m in grp])
        zdif=(zmax[0]-zmin[0])
        print(zdif)
        print(zmin)
        return train_df, valid_df, train_df1, train_df11, test_df0, test_df11, grp, zmin, zmax

    def load_weights(self, model, model_path):
        state = torch.load(str(model_path))
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state)

        return model

    def model_train(self, train_df, valid_df, BATCH_SIZE=32, lr=1e-2, n_epochs=60, model_path=None):

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

        if not model_path:
            if torch.cuda.is_available():
                model.cuda()        
            train5(init_optimizer=lambda lr: torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001),#(model.parameters(), lr=lr),
                    lr = lr,#1e-4
                    n_epochs = n_epochs,#35,#40,
                    model=model,
                    criterion=criterionL5,#LossBinaryElu(jaccard_weight=0),
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    validation=validation5,
                 )
        else:
            model = load_weights(model, model_path)
            if torch.cuda.is_available():
                model.cuda()
            retrain5(init_optimizer=lambda lr: torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001),#(model.parameters(), lr=lr),
            lr = lr,#1e-4
            n_epochs = n_epochs,#35,#40,
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

        plt.xlabel('step')
        plt.legend(loc='center left')
        plt.tight_layout()
        plt.show();

    def determine_thrsh(self, valid_df, train_df1, model_path ='model_1.pt', batchsize=16):
        model = LinkNet34deeps(1)

        if ('id' in train_df1.columns.values.tolist()):
            train_df1=train_df1.set_index('id')

        state = torch.load(str(model_path))
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state)
        if torch.cuda.is_available():
            model.cuda()

        model.eval()

        loader = DataLoader(
                dataset=SaltDataset(valid_df, transform=None, mode='valid'),
                shuffle=False,
                batch_size=batchsize,#BATCH_SIZE,
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
        return threshold_best


    ##test dataset########################################################
    def model_predict(self, test_image_dir, test_df0, test_df11, threshold_best)
        test_paths = os.listdir(test_image_dir+'/images/')
        print(len(test_paths), 'test images found')

        test_df0['rle_mask'] = None

        train_df=0
        train_df0=0
        train_df1=0
        grp=0

        if not('id' in test_df0.columns.values.tolist()):
            test_df0.reset_index(inplace=True)

        #from skimage.morphology import binary_opening, disk
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

        sub = self.generate_submission(pred_dict) 

        return preds_dict, sub

    def generate_submission(self, pred_dict)              
        sub = pd.DataFrame.from_dict(pred_dict,orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('submissionpth.csv')
        return sub
#######################################################################################################################################

def main(config, aug_list):
    if config.mode == 'train':
        solver = ModelSolver(config)
        solver.train_fold(config.train_fold_index, aug_list)
    if config.mode == 'test':
        solver = ModelSolver(config)
        solver.infer_fold_all_Cycle(config.train_fold_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_fold_index', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)

    aug_list = ['flip_lr']
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--pretrained_model', type=str, default='model_1.pt')

    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--epochs', type=int, default=10)

    # Test settings
    parser.add_argument('--traincsv_path', type=str, default='train.csv')
    parser.add_argument('--depthscsv_path', type=str, default='depths.csv')
    parser.add_argument('--train_image_path', type=str, default='train/images')
    parser.add_argument('--train_mask_path', type=str, default='train/masks')
    parser.add_argument('--test_image_dir', type=str, default='test')

    config = parser.parse_args()
    print(config)
    main(config, aug_list)

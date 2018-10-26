def dummy_prediction(X_test,model,mirror=True):
    #X_test0 = variable(X_test, volatile=True)
    # ...
    y_preds = model(X_test)
    y_preds = F.sigmoid(y_preds).data.cpu().numpy()
    if mirror==True:
        #print(X_test.shape)#32-3-128-128
        m_preds = model(variable(torch.FloatTensor((X_test.cpu().data.numpy()[:,:,:,::-1]).copy()), volatile=True))
        ##m_preds = model(Variable(torch.from_numpy(X_test.cpu().data.numpy()[:,:,::-1,:])))
        m_preds = F.sigmoid(variable(torch.FloatTensor((m_preds.cpu().data.numpy()[:,:,:,::-1]).copy()), volatile=True)).data.cpu().numpy()
        y_preds = 0.5 * (y_preds + m_preds)
        #y_preds = 0.5 * (y_preds + variable(torch.FloatTensor((m_preds.cpu().data.numpy()[:,:,::-1,:]).copy()), volatile=True))
    return y_preds
def dummy_prediction4(X_test,model,mirror=True):
    #X_test0 = variable(X_test, volatile=True)
    # ...
    y_preds,o2,o3,o4 = model(X_test)
    y_preds = F.sigmoid(y_preds).data.cpu().numpy()
    if mirror==True:
        #print(X_test.shape)#32-3-128-128
        m_preds,o2,o3,o4 = model(variable(torch.FloatTensor((X_test.cpu().data.numpy()[:,:,:,::-1]).copy()), volatile=True))
        ##m_preds = model(Variable(torch.from_numpy(X_test.cpu().data.numpy()[:,:,::-1,:])))
        m_preds = F.sigmoid(variable(torch.FloatTensor((m_preds.cpu().data.numpy()[:,:,:,::-1]).copy()), volatile=True)).data.cpu().numpy()
        y_preds = 0.5 * (y_preds + m_preds)
        #y_preds = 0.5 * (y_preds + variable(torch.FloatTensor((m_preds.cpu().data.numpy()[:,:,::-1,:]).copy()), volatile=True))
    return y_preds
def dummy_prediction2(X_test,model,mirror=True):
    #X_test0 = variable(X_test, volatile=True)
    # ...
    y_preds,o2,o3,o4,o5 = model(X_test)
    y_preds = F.sigmoid(y_preds).data.cpu().numpy()
    if mirror==True:
        #print(X_test.shape)#32-3-128-128
        m_preds,o2,o3,o4,o5 = model(variable(torch.FloatTensor((X_test.cpu().data.numpy()[:,:,:,::-1]).copy()), volatile=True))
        ##m_preds = model(Variable(torch.from_numpy(X_test.cpu().data.numpy()[:,:,::-1,:])))
        m_preds = F.sigmoid(variable(torch.FloatTensor((m_preds.cpu().data.numpy()[:,:,:,::-1]).copy()), volatile=True)).data.cpu().numpy()
        y_preds = 0.5 * (y_preds + m_preds)
        #y_preds = 0.5 * (y_preds + variable(torch.FloatTensor((m_preds.cpu().data.numpy()[:,:,::-1,:]).copy()), volatile=True))
        
    return y_preds









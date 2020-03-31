################################################################
##      ##   #            #####       #         #######  #   #
# #    # #   #            #   #      # #       #         #  #
#  #  #  #   #            #####     #   #     #          # #
#   #    #   #            #        # # # #    #          ###
#        #   #            #       #       #    #         #  # 
#        #   #######      #      #         #    #######  #    #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


################################################## K-Means Clustering for 1D data ############################################
def KMeansClusterer1D(values, n, init_mu, n_iter = 10 ):
    
    '''
	
	Performs K-Means Clustering on uni-dimensional data.
    values : 1D Vector containing values to cluster
    n : Number of clusters to make
    init_mu : list containing 'n' values for initial centres
    
    RETURNS:
    new_mu : list of new clusters location
    radius : list of cluster radius
    '''
    
    assert len(init_mu) == n
    
    
    for i in range(n_iter):
        
        val_mu_diff = np.reshape(np.power(values - init_mu[0], 2), [-1,1])
        #To calculate x_i - mu_c 
        ## and
        #To calculate cluster which minimizes square of val_mu_diff
        for j in range(1,n):
            val_mu_diff = np.concatenate([ val_mu_diff, np.reshape(np.power(values - init_mu[j], 2), [-1,1]) ], axis = 1)
            
        c = np.argmin(val_mu_diff, axis = 1)
        
        #Updating Centroids
        new_mu = []
        for j in range(n):
            
            temp = values[c == j]
            new_mu.append(np.sum(temp) / len(temp))
        
        init_mu = new_mu
        
    
    #To calculate Radius of each cluster
    radius = []
    for j in range(n):
        temp = values[c == j]
        radius.append(np.abs(np.max(temp) - new_mu[j]))        
        
    return new_mu, radius

############################################### Complete Decision Trees #######################################################
################################################################
##      ##   #            #####       #         #######  #   #
# #    # #   #            #   #      # #       #         #  #
#  #  #  #   #            #####     #   #     #          # #
#   #    #   #            #        # # # #    #          ###
#        #   #            #       #       #    #         #  # 
#        #   #######      #      #         #    #######  #    #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


################################################## K-Means Clustering for 1D data ############################################
def KMeansClusterer1D(values, n, init_mu, n_iter = 10 ):
    
    '''
	
	Performs K-Means Clustering on uni-dimensional data.
    values : 1D Vector containing values to cluster
    n : Number of clusters to make
    init_mu : list containing 'n' values for initial centres
    
    RETURNS:
    new_mu : list of new clusters location
    radius : list of cluster radius
    '''
    
    assert len(init_mu) == n
    
    
    for i in range(n_iter):
        
        val_mu_diff = np.reshape(np.power(values - init_mu[0], 2), [-1,1])
        #To calculate x_i - mu_c 
        ## and
        #To calculate cluster which minimizes square of val_mu_diff
        for j in range(1,n):
            val_mu_diff = np.concatenate([ val_mu_diff, np.reshape(np.power(values - init_mu[j], 2), [-1,1]) ], axis = 1)
            
        c = np.argmin(val_mu_diff, axis = 1)
        
        #Updating Centroids
        new_mu = []
        for j in range(n):
            
            temp = values[c == j]
            new_mu.append(np.sum(temp) / len(temp))
        
        init_mu = new_mu
        
    
    #To calculate Radius of each cluster
    radius = []
    for j in range(n):
        temp = values[c == j]
        radius.append(np.abs(np.max(temp) - new_mu[j]))        
        
    return new_mu, radius

############################################### Complete Decision Trees #######################################################

class DecisionTrees:

    
    """
    ***************DECISION TREES********************
    
    Call function TreeMaker() to get the Root of the tree with all connections to 
    next branches.
    """
     
    def __init__(self, dataset, max_depth, min_samples, n_classes):
        
        self.dataset = dataset
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_classes = n_classes
        
        
    
    class TreeNode:
    
        def __init__(self, data, NodeL, NodeR, n_classes):

            assert data.ndim == 2

            self.data = data
            self.values = np.zeros(n_classes)
            for i in range(n_classes):
                self.values[i] = np.sum(self.data[:,-1] == i)
            self.NodeL = NodeL
            self.NodeR = NodeR
            self.giniScore = self.NodeGinifromData(self.data, n_classes)[0]
            self.threshold = None
            if (((self.NodeL != None) or (self.NodeR!= None)) and self.threshold == None):
                raise ValueError("Node's Threshold not set before splitting.")
                
        def NodeGinifromData(self, data, n_classes):
        
            """
            Calculates Gini Score for a Node from the data directly

            data : Array of shape = [n_rows, n_features]
            """

            values = np.zeros([n_classes])

            for i in range(n_classes):
                values[i] = np.sum(data[data[:,-1] == i])

            return self.NodesGiniScore([values])
        
        
        
        def NodesGiniScore(self,arrNodes):
            """
            Takes a Numpy Array containing Node's no. of class instances for each class
            & returns each Node's Gini Score

            arrNodes : shape = [No. of Nodes, No. of Classes], np.array type containing of Nodes with each 
                        element containing the number of class instances

            RETURNS : 

            Vector containing the Gini score of each Node that was passed 

            Example:

            IN>>NodesGiniScore(np.array([[1,1,34],[1,2,3], [50,50,50]]))
            OUT>>array([0.10648148, 0.61111111, 0.66666667])


            """

            Nodesums = np.sum(arrNodes, axis = 1)

            Nodesums = Nodesums.astype(np.float64)

            Nodesums[Nodesums == 0] = np.inf

            return 1 - np.sum(np.power(arrNodes*np.reshape((1/Nodesums),[-1,1]),2), axis = 1)

        
    
    
        
    
    
    def NodesGiniScore(self,arrNodes):
        """
        Takes a Numpy Array containing Node's no. of class instances for each class
        & returns each Node's Gini Score

        arrNodes : shape = [No. of Nodes, No. of Classes], np.array type containing of Nodes with each 
                    element containing the number of class instances

        RETURNS : 

        Vector containing the Gini score of each Node that was passed 

        Example:

        IN>>NodesGiniScore(np.array([[1,1,34],[1,2,3], [50,50,50]]))
        OUT>>array([0.10648148, 0.61111111, 0.66666667])


        """

        Nodesums = np.sum(arrNodes, axis = 1)

        Nodesums = Nodesums.astype(np.float64)

        Nodesums[Nodesums == 0] = np.inf

        return 1 - np.sum(np.power(arrNodes*np.reshape((1/Nodesums),[-1,1]),2), axis = 1)
    
    
    def Split_GiniIndex(self, splitL, splitR, n_classes):
        """
        Calculates the Gini Index (or the Cost for the split) for the 
        split nodes entered.

        splitL, splitR : Arrays of shape = [No. of Rows for this split, No. of features], the array after splitting the dataset
        n_classes : The number of classes.

        NOTE: 1. Function expects for the last feature/column to contain the Class Index
                 in the numerical form, i.e. like 0,1,2,...,n-1

              2. This function also neglects that split which is empty

        RETURNS:

        The final Gini Index for this Split.
        """


        valuesL = np.zeros(n_classes)
        valuesR = np.zeros(n_classes)

        if splitL.shape[0] == 0:
            for i in range(n_classes):
                valuesR[i] = np.sum(splitR[:,-1] == i)
        elif splitR.shape[0] == 0:
            for i in range(n_classes):
                valuesL[i] = np.sum(splitL[:,-1] == i)
        else:
            for i in range(n_classes):
                valuesR[i] = np.sum(splitR[:,-1] == i)
                valuesL[i] = np.sum(splitL[:,-1] == i)



        GiniL, GiniR = self.NodesGiniScore([valuesL, valuesR])

        Lrows = splitL.shape[0]
        Rrows = splitR.shape[0]

        GiniIndex = GiniL * Lrows/(Lrows+Rrows) + GiniR * Rrows/(Lrows+Rrows) 
    
        return GiniIndex
    

    def Splitter(self, values ,threshold, feature_index = 0):
        """
        Splits the dataset (values) into two subsets via splitting through threshold
        for feature present at feature_index

        values : shape = [No. of rows, No. of features], The dataset to split
        threshold : Float, the value by which to compare and split the dataset
        feature_index : Integer, the index of the feature(column) through which
                        comparison will be drawn for splitting

        RETURNS : 

        The two splits done by threshold for feature_index.
        """
        if feature_index >= values.shape[1]-1:
            raise ValueError("feature_index {} is greater the possible column index values {}. Note that it's assumed that last column is for numerical class ".format(feature_index,values.shape[1]))

        left = values[values[:,feature_index] <= threshold]
        right = values[values[:,feature_index] > threshold]

        return left, right
    
    
    def SplitEvaluator(self, dataset, n_classes):
        """
        Splits the dataset by each value in each attribute and then 
        finds the best split via Gini Impurity.

        dataset : Numpy array of shape = [No. of Rows, No. of features], the data itself in raw form.

        RETURNS : 

        Returns the Matrix containing gini Index value for the split using that row's 
        feature and threshold and index of best split in this Matrix (thus, the dataset too.)

        giniIndexes : Shape = [dataset.shape[0], dataset.shape[1]-1]
        (row, feat) : Coordinates for the best split in dataset or giniIndexes.
        threshold and the feature index of the threshold.
        """
        giniIndexes = np.zeros([dataset.shape[0], dataset.shape[1]-1]) #-1 to remove the class feature

        for feat_i in range(dataset.shape[1] - 1):

            for row_i in range(dataset.shape[0]):

                threshold = dataset[row_i,feat_i]

                splitL, splitR = self.Splitter(dataset, threshold, feat_i)

                #Calculate this split's Gini Index

                giniInd = self.Split_GiniIndex(splitL, splitR, n_classes)

                giniIndexes[row_i, feat_i] = giniInd

        coords =  np.argwhere(giniIndexes == np.min(giniIndexes))

        bestsplitL, bestsplitR = self.Splitter(dataset, dataset[coords[0,0], coords[0,1]], coords[0,1])
        best_gini = giniIndexes[coords[0,0], coords[0,1]]
        threshold = dataset[coords[0,0], coords[0,1]], coords[0,1]

        return giniIndexes, threshold, best_gini, bestsplitL, bestsplitR

#{'GiniIndex':best_gini, 'NodeL': {'data': bestsplitL, 'NodeL':None, 'NodeR':None},\
                        #'NodeR': {'data': bestsplitR, 'NodeL':None, 'NodeR':None}  } 
    
    def NodeGinifromData(self, data, n_classes):
        
        """
        Calculates Gini Score for a Node from the data directly

        data : Array of shape = [n_rows, n_features]
        """

        values = np.zeros([n_classes])

        for i in range(n_classes):
            values[i] = np.sum(data[data[:,-1] == i])

        return self.NodesGiniScore([values])
    
    
    def LeafClass(self, LeafValues):
        """
        Returns the final class for the Leaf Node

        LeafValues : Array of shape = [no. of classes], No. of instances of each class
                     in the leaf node.

        RETURNS:

        Class to which this Leaf Node belongs to.
        """

        return np.argmax(LeafValues)
    
    def TreeNodeFromNode(self, Node, n_classes):
        """
        Function to take input a data and find the best split and Add two nodes 
        in the Node

        Node : An instance of TreeNode class
        n_classes : Integer, no. of classes

        RETURNS :

        Node after adding two nodes L and R after purest split.
        """

        data = Node.data
        _,thres,_,dataL, dataR = self.SplitEvaluator(data, n_classes)
        NodeL = self.TreeNode(dataL, None, None, n_classes)
        NodeR = self.TreeNode(dataR, None, None, n_classes)

        Node.threshold = thres
        Node.NodeL = NodeL
        Node.NodeR = NodeR

        return None
    
        
    def RecursiveTreeMaker(self, ListOfNodes, max_depth, min_samples, curr_depth, n_classes ): 
        """
        Makes the tree recursively
        
        """


        if curr_depth > max_depth:
            return ListOfNodes

        newListOfNodes = []
        for node in ListOfNodes:

            if node.giniScore == 0.0:
                continue

            elif np.sum(node.values) < min_samples:
                continue 

            self.TreeNodeFromNode(node, n_classes)

            newListOfNodes.append(node.NodeL)
            newListOfNodes.append(node.NodeR)

        print(len(newListOfNodes))

        self.RecursiveTreeMaker(newListOfNodes, max_depth, min_samples, curr_depth+1, n_classes)

        
    def TreeMaker(self):
        """
        Creates the Tree and returns it's Root Node

        dataset : Array of shape = [n_rows, n_feat]
        max_depth : Maximum depth of the tree
        min_samples : Minimum no. of samples to be present at a Node,
                        otherwise node wouldn't be constructed.
        n_classes : No.of classes
        """

        Root = self.TreeNode(self.dataset, None, None, self.n_classes)

        self.RecursiveTreeMaker([Root], self.max_depth, self.min_samples, 1 , self.n_classes)

        return Root  

    

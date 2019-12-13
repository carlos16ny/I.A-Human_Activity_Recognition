class Data:

    def __init__(self):
        self.classes = []
        self.X_train = []
        self.X_test  = []
        self.Y_train = []
        self.Y_test  = []
        self.loadData()
    
    def loadData(self):

        files = {
            'X_test'  : 'test/X_test.txt',
            'Y_test'  : 'test/y_test.txt',
            'X_train' : 'train/X_train.txt',
            'Y_train' : 'train/y_train.txt',
            'classes' : './activity_labels.txt'
        }

        with open(files['classes'], 'r') as classes:
            self.classes = [x.split("\n")[0] for x in classes.readlines()]
        
        with open(files['X_test'], 'r') as xtest:
            for line in xtest:
                linha = []
                for number in line.strip().split():
                    linha.append(float(number))
                self.X_test.append(linha)

        with open(files['X_train'], 'r') as xtrain:
            for line in xtrain:
                linha = []
                for number in line.strip().split():
                    linha.append(float(number))
                self.X_train.append(linha)

        with open(files['Y_test'], 'r') as ytest:
            for line in ytest:
                self.Y_test.append(line.split("\n")[0])

        with open(files['Y_train'], 'r') as ytrain:
            for line in ytrain:
                self.Y_train.append(line.split("\n")[0])
                

            
            


        

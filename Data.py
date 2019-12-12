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
            'Y_train' : 'train/Y_train.txt',
            'classes' : './activity_labels.txt'
        }

        with open(files['classes'], 'r') as classes:
            self.classes = [x.split("\n")[0] for x in classes.readlines()]
        
        with open(files['X_test'], 'r') as xtest:
            lines = xtest.readlines()
            for line in lines:
                linha = []
                for l in line:
                    linha.append(l)

                self.X_test.append(linha)
    
    def printLoad(self):
        # print(self.X_test)
        print(self.classes)



data = Data()
data.printLoad()
            
            


        

import math
import random

from Tree import DecicisonTree
from BNC import BNC
class IntegratedLearning:
    def __init__(self):
        self.data = []
        #子学习器数量
        self.LearnersNum = 0
        #基础学习器
        self.ComponentLearners = []
        #学习器权重
        self.LearnerWeightArray = []
        #标签(保证只有两个类别，正类定义为1，负类定义为-1)
        self.label = []
        #权值数组
        self.DataWeightArray = []

    def Train(self, LearnersNum = 3, LearnerType = BNC, data = [], label = []):
        self.data = data
        self.label = label
        for index in range(len(data)):
            self.DataWeightArray.append(1/len(data))
        for index in range(LearnersNum):
            Epre = 1/2 * (math.exp(1) + math.exp(-1))
            if self.LearnersNum != 0:
                #计算当前的期望
                tot = 0
                for i in range(len(data)):
                    predict = self.Predict(data[i])
                    tot += math.exp(1-2*(predict == label[i]))
                Epre = tot/len(data)
            #构建学习器
            learner = LearnerType()
            #训练数据
            #重采样生成新的数据集
            ResampleData = []
            ResampleLabel = []
            tot = 0
            for i in range(len(self.DataWeightArray)):
                tot += self.DataWeightArray[i]
            while True:
                pos = random.randint(0, len(data)-1)
                pro = random.uniform(0, 1)
                if pro >= self.DataWeightArray[pos]:
                    ResampleData.append(data[pos])
                    ResampleLabel.append(label[pos])
                if len(ResampleData) == len(data) * 10:
                    break
            learner.Train(data=ResampleData, label=ResampleLabel)
            #计算误差
            #总量
            tot = 0
            #误差
            diff = 0
            for i in range(len(data)):
                tot += self.DataWeightArray[i]
                predict = learner.Predict(data[i])
                if predict != label[i]:
                    diff += self.DataWeightArray[i]
            diff /= tot

            if diff > 0.5:
                break
            self.ComponentLearners.append(learner)
            self.LearnersNum += 1
            if diff == 0:
                self.LearnerWeightArray.append(float('inf'))
                break
            self.LearnerWeightArray.append(1 / 2 * math.log((1 - diff) / diff))
            tot = 0
            for i in range(len(data)):
                predict = self.Predict(data[i])
                tot += math.exp(1 - 2 * (predict == label[i]))
            Enow = tot / len(data)
            for i in range(len(data)):
                predict = learner.Predict(data[i])
                self.DataWeightArray[i] = self.DataWeightArray[i] * \
                                          math.exp(diff*(1-2*(label[i] == predict))) \
                                          * Epre/Enow

    def Predict(self, data):
        res = {}
        for index in range(self.LearnersNum):
            predict = self.ComponentLearners[index].Predict(data)
            if predict not in res.keys():
                res[predict] = 0
            res[predict] += self.LearnerWeightArray[index]
        totPredict = None
        num = 0
        for key in res.keys():
            if res[key] > num:
                num = res[key]
                totPredict = key
        return totPredict

    def Score(self, data=[], label=[]):
        tot = 0
        for index in range(len(data)):
            tot += self.Predict(data[index]) == label[index]
        return tot / len(data)


if __name__ == "__main__":
    datas = [["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
             ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
             ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
             ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
             ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
             ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘"],
             ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘"],
             ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑"],

             ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑"],
             ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘"],
             ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑"],
             ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘"],
             ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑"],
             ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑"],
             ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘"],
             ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑"],
             ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑"]]

    labels = ["好瓜", "好瓜", "好瓜", "好瓜", "好瓜", "好瓜", "好瓜", "好瓜",
              "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜"]
    il = IntegratedLearning()
    il.Train(LearnersNum=5, data=datas, label=labels, LearnerType=DecicisonTree)
    print(il.Predict(["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑"]))
    print(il.Score(data=datas, label=labels))

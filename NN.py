 # -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:30:07 2017

@author: MohamedIsmail
"""
import tkinter as Tk
import numpy as np
from tkinter import *
import pandas as pd
import math as m
class Design:
    def __init__(self):
        self.root=Tk()
        self.root.geometry("700x500")
        self.var1=IntVar()
        self.t1= Entry(self.root)
        self.t2= Entry(self.root)
        self.t3= Entry(self.root)
        self.t4= Entry(self.root)
        self.t5= Entry(self.root)
        self.t6= Entry(self.root)
        self.t7= Entry(self.root)
        self.t8= Entry(self.root)
        self.t9= Entry(self.root)
        self.bias=IntVar()
        self.flag=0
        self.items()
        self.root.mainloop()
         
    def items(self):
        labl1=Label(self.root,text="Enter the values:",font=(18)).grid(row=0)
        labl2=Label(self.root,text="No. Of epochs",font=(10)).grid(row=1)
        self.t1.grid(row=1,column=1)
        labl3=Label(self.root,text="Learning rate",font=(10)).grid(row=2)
        self.t2.grid(row=2,column=1)
        labl4=Label(self.root,text="Enter the number of MultiLayers:",font=(10)).grid(row=3)
        self.t3.grid(row=3,column=1)
        labl5=Label(self.root,text="Enter the number of nodes in each layer:",font=(10)).grid(row=4)
        self.t4.grid(row=4,column=1)
        labl6=Label(self.root,text="Choose Type of function:",font=(10)).grid(row=5)
        radio1=Radiobutton(self.root,text="Segmoid",command=self.option1,value=1).grid(row=6,sticky=W)
        radio1=Radiobutton(self.root,text="Hyperbolic",command=self.option2,value=2).grid(row=7,sticky=W)
        btn=Button(self.root,text="Okey",command=self.TestingData).grid(row=2,column=2,padx=20,pady=5)
        #btn=Button(self.root,text="read",command=self.readFeatures).grid(row=3,column=2,padx=20,pady=5)
        checkbox = Checkbutton(self.root, text="Add bias", variable=self.bias).grid(row=6,column =0)
        labl7=Label(self.root,text="The Accuracy:",font=(10)).grid(row=8)
        self.t5.grid(row=8,column=1)
        labl8=Label(self.root,text="Enter The Features:",font=(10)).grid(row=9)
        labl9=Label(self.root,text="Feature1: ",font=(6)).grid(row=10)
        self.t6.grid(row=10,column=1)
        labl10=Label(self.root,text="Feature2: ",font=(6)).grid(row=11)
        self.t7.grid(row=11,column=1)
        labl11=Label(self.root,text="Feature3: ",font=(6)).grid(row=12)
        self.t8.grid(row=12,column=1)
        labl12=Label(self.root,text="Feature4: ",font=(6)).grid(row=13)
        self.t9.grid(row=13,column=1)
        btn=Button(self.root,text="Test",command=self.testOneFlower).grid(row=14,column=1,padx=20,pady=5)
     
    def option1(self):
        self.flag=1
        
    def option2(self):
        self.flag=2
        
    def prnt(self):
        l=[]
        l.append(int(self.t1.get()))
        l.append(float(self.t2.get()))
        l.append(int(self.t3.get()))
        l.append(int(self.t4.get()))
        return l
    def readFeatures(self):
        l=[]
        l.append(float(self.t6.get()))
        l.append(float(self.t7.get()))
        l.append(float(self.t8.get()))
        l.append(float(self.t9.get()))
        return l
    
    def readData(self):
        df=pd.read_csv("Iris Data.csv")
        l=self.prnt()
        matrixInputs=np.zeros((max(l[3],4),90))
        matrixInputs[:][0]=df['X1'][:90].copy()
        matrixInputs[:][1]=df['X2'][:90].copy()
        matrixInputs[:][2]=df['X3'][:90].copy()
        matrixInputs[:][3]=df['X4'][:90].copy()
        return matrixInputs
    
    def readDataTesting(self):
        df=pd.read_csv("Iris Data.csv")
        matrixInputsTesting=np.zeros((4,60))
        matrixInputsTesting[:][0]=df['X1'][90:150]
        matrixInputsTesting[:][1]=df['X2'][90:150]
        matrixInputsTesting[:][2]=df['X3'][90:150]
        matrixInputsTesting[:][3]=df['X4'][90:150]
        return matrixInputsTesting
    
    def initialWeights(self):
        l=[]
        l=self.prnt()
        if self.bias.get():
            matrix=np.random.rand(l[2],l[3],max(l[3],5))
        else:
            matrix=np.random.rand(l[2],l[3],max(l[3],4))
        return matrix

    def feedForwardAndBackward(self):
        l=[]
        l=self.prnt()
        biasRandom=np.random.rand(1)
        matrixInputs=self.readData()
        #print(matrixInputs)
        matrixWeights=self.initialWeights()
        #print(matrixWeights)
        yActivation=np.zeros((l[2],l[3]))
        errors=np.zeros((l[2],l[3]))
        #print(errors)
        #print(l[2])
        for epochs in range(l[0]):
            for k in range(90):
                for p in range(l[2]):
                    if p==0:
                        for i in range(l[3]):
                            netInput=0
                            for j in range(4):
                                if self.bias.get():
                                    netInput+=(matrixWeights[0][i][j]*matrixInputs[j][k])+biasRandom
                                else:
                                    netInput+=matrixWeights[0][i][j]*matrixInputs[j][k]
                            yActivation[p][i]=self.yFunction(netInput)   
                    elif p==l[2]-1:
                        for i in range(3):
                            netInput=0
                            for j in range(l[3]):
                                if self.bias.get():    
                                    netInput+=(matrixWeights[p][i][j]*yActivation[p-1][j])+biasRandom
                                else:
                                    netInput+=matrixWeights[0][i][j]*yActivation[p-1][j]
                            yActivation[p][i]=self.yFunction(netInput)
                    else:
                        for i in range(l[3]):
                            netInput=0
                            for j in range(l[3]):
                                if self.bias.get():
                                    netInput+=(matrixWeights[p][i][j]*yActivation[p-1][j])+biasRandom
                                else:
                                    netInput+=matrixWeights[p][i][j]*yActivation[p-1][j]
                            yActivation[p][i]=self.yFunction(netInput)
                x=int(k/30)
                if x== 0:
                    errors[l[2]-1][0]=(1-yActivation[l[2]-1][0])*self.derivitiveOfYfunction(yActivation[l[2]-1][0])
                    errors[l[2]-1][1]=(0-yActivation[l[2]-1][1])*self.derivitiveOfYfunction(yActivation[l[2]-1][1])
                    errors[l[2]-1][2]=(0-yActivation[l[2]-1][2])*self.derivitiveOfYfunction(yActivation[l[2]-1][2])
                    for i in range(3):
                        for j in range(l[3]):        
                            matrixWeights[l[2]-1][i][j]=matrixWeights[l[2]-1][i][j]+(l[1]*errors[l[2]-1][i]*yActivation[l[2]-1][i])
                elif x==1:
                    errors[l[2]-1][0]=(0-yActivation[l[2]-1][0])*self.derivitiveOfYfunction(yActivation[l[2]-1][0])
                    errors[l[2]-1][1]=(1-yActivation[l[2]-1][1])*self.derivitiveOfYfunction(yActivation[l[2]-1][1])
                    errors[l[2]-1][2]=(0-yActivation[l[2]-1][2])*self.derivitiveOfYfunction(yActivation[l[2]-1][2])
                    for i in range(3):
                        for j in range(l[3]):        
                            matrixWeights[l[2]-1][i][j]=matrixWeights[l[2]-1][i][j]+(l[1]*errors[l[2]-1][i]*yActivation[l[2]-1][i])
                else:
                    errors[l[2]-1][0]=(0-yActivation[l[2]-1][0])*self.derivitiveOfYfunction(yActivation[l[2]-1][0])
                    errors[l[2]-1][1]=(0-yActivation[l[2]-1][1])*self.derivitiveOfYfunction(yActivation[l[2]-1][1])
                    errors[l[2]-1][2]=(1-yActivation[l[2]-1][2])*self.derivitiveOfYfunction(yActivation[l[2]-1][2])
                    for i in range(3):
                        for j in range(l[3]):        
                            matrixWeights[l[2]-1][i][j]=matrixWeights[l[2]-1][i][j]+(l[1]*errors[l[2]-1][i]*yActivation[l[2]-1][i])
                for p in range(l[2]-1,l[2]-2,-1):
                    for i in range(l[3]):
                        errors[p-1][i]=0
                        for j in range(3):
                            errors[p-1][i]+=errors[p][j]*matrixWeights[p][j][i]*self.derivitiveOfYfunction(yActivation[p][j])
                for p in range(l[2]-2,0,-1):
                    for i in range(l[3]):
                        errors[p-1][i]=0
                        for j in range(l[3]):
                            errors[p-1][i]+=errors[p][j]*matrixWeights[p][j][i]*self.derivitiveOfYfunction(yActivation[p][j])
                for p in range(l[2]):
                    for i in range(l[3]):
                        if p ==0:
                            for j in range(4):
                                matrixWeights[p][i][j]= matrixWeights[p][i][j]+(l[1]*errors[p][i]*matrixInputs[j][k])
                        else:
                            for j in range(l[3]):
                                matrixWeights[p][i][j]=matrixWeights[p][i][j]+(l[1]*errors[p][i]*matrixInputs[j][k])
        
        return matrixWeights
        
    def TestingData(self):
        matrixInputsTesting=self.readDataTesting()
        l=self.prnt()
        biasRandom=np.random.rand(1)
        updatedWeights=self.feedForwardAndBackward()
        yActivation=np.zeros((l[2],l[3]))
        confusionMatrix=np.zeros((3,3))
        for k in range(60):
            for p in range(l[2]):
                if p==0:
                    for i in range(l[3]):
                        netInput=0
                        for j in range(4):
                            if self.bias.get():
                                netInput+=updatedWeights[0][i][j]*matrixInputsTesting[j][k]+biasRandom
                            else:
                                netInput+=updatedWeights[0][i][j]*matrixInputsTesting[j][k]
                        yActivation[p][i]=self.yFunction(netInput)
                elif p==l[2]-1:
                    for i in range(3):
                        netInput=0
                        for j in range(l[3]):
                             if self.bias.get():    
                                    netInput+=(updatedWeights[p][i][j]*yActivation[p-1][j])+biasRandom
                             else:
                                netInput+=updatedWeights[0][i][j]*yActivation[p-1][j]
                        yActivation[p][i]=self.yFunction(netInput)
                else:
                    
                    for i in range(l[3]):
                        netInput=0
                        for j in range(l[3]):   
                            if self.bias.get():
                                netInput+=(updatedWeights[p][i][j]*yActivation[p-1][j])+biasRandom
                            else:
                                netInput+=updatedWeights[p][i][j]*yActivation[p-1][j]
                        yActivation[p][i]=self.yFunction(netInput)
            x=int(k/20)
            y=max(yActivation[l[2]-1][0],yActivation[l[2]-1][1],yActivation[l[2]-1][2])
            z=0
            if y == yActivation[l[2]-1][0]:
                z=0
                confusionMatrix[x][z]+=1
            elif y == yActivation[l[2]-1][1]:
                z=1
                confusionMatrix[x][z]+=1
            elif y == yActivation[l[2]-1][2]:
                z=2
                confusionMatrix[x][z]+=1
        
        print(confusionMatrix)
        print("The Accuracy= ",((confusionMatrix[0][0]+confusionMatrix[1][1]+confusionMatrix[2][2])/60)*100,"%")
        return
    def testOneFlower(self):
        l1=self.prnt()
        biasRandom=np.random.rand(1)
        matrixInputsFeatures=self.readFeatures()
        updatedWeights=self.feedForwardAndBackward()
        yActivation=np.zeros((l1[2],l1[3]))
        for p in range(l1[2]):
                if p==0:
                    for i in range(l1[3]):
                        netInput=0
                        for j in range(4):
                            if self.bias.get():
                                netInput+=updatedWeights[0][i][j]*matrixInputsFeatures[j]+biasRandom
                            else:
                                netInput+=updatedWeights[0][i][j]*matrixInputsFeatures[j]
                        yActivation[p][i]=self.yFunction(netInput)
                elif p==l1[2]-1:
                    for i in range(3):
                        netInput=0
                        for j in range(l1[3]):
                             if self.bias.get():    
                                    netInput+=(updatedWeights[p][i][j]*yActivation[p-1][j])+biasRandom
                             else:
                                netInput+=updatedWeights[0][i][j]*yActivation[p-1][j]
                        yActivation[p][i]=self.yFunction(netInput)
                else:
                    
                    for i in range(l1[3]):
                        netInput=0
                        for j in range(l1[3]):   
                            if self.bias.get():
                                netInput+=(updatedWeights[p][i][j]*yActivation[p-1][j])+biasRandom
                            else:
                                netInput+=updatedWeights[p][i][j]*yActivation[p-1][j]
                        yActivation[p][i]=self.yFunction(netInput)
        y=max(yActivation[l1[2]-1][0],yActivation[l1[2]-1][1],yActivation[l1[2]-1][2])
        if y == yActivation[l1[2]-1][0]:
            print('Setosa')
        elif y == yActivation[l1[2]-1][1]:
            print('VersiColor')
        elif y == yActivation[l1[2]-1][2]:
            print('virginica')
                
    def yFunction(self,vk):
        if self.flag==1:
            try:
                x=1/(1+m.exp(-vk))
            except OverflowError:
                x=float('inf')
            except ZeroDivisionError:
                x = float('Inf')
        else:
            try:    
                x=(1-m.exp(-vk)/(1+m.exp(-vk)))
            except OverflowError:
                x=float('inf')
            except ZeroDivisionError:
                x = float('Inf')
        return x
    
    def derivitiveOfYfunction(self,vk):
        if self.flag==1:
            try:    
                x=self.yFunction(vk)*(1-self.yFunction(vk))
            except OverflowError:
                x=float('inf')
            except ZeroDivisionError:
                x = float('Inf')
        else:
            try:    
                x=m.exp(-vk)/(1+m.exp(-vk)*m.exp(-vk))
            except OverflowError:
                x=float('inf')
            except ZeroDivisionError:
                x = float('Inf')
        return x
run=Design()
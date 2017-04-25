from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
from scipy.spatial.distance import dice
import numpy as np
import os
# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    #setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    #setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    #setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')


labelnum=14
fold='../res/norandomrun_0'
#fold='norandom_1_noquicktest'
resdir=fold+"/"
objectNum=25


if os.path.exists(fold+'_dice.npy'):
    precision=np.load(fold+'_acc.npy')
    recall=np.load(fold+'_rec.npy')
    dice=np.load(fold+'_dice.npy')
else:
    precision=[]
    recall=[]
    dice=[]

    for i in range(labelnum):
        precision.append([])
        recall.append([])
        dice.append([])
    for objectInd in range(objectNum):
        seg=np.load(resdir+'%d_seg.npy'%(objectInd))
        vote=np.load(resdir+'%d_vote.npy'%(objectInd))
        seg=seg.flatten()
        vote=vote.flatten()
        allcorrect=(seg==vote)
        print objectInd,"/objectNum"
        for i in range(labelnum):
            labelind=i+1
            # other
            if labelind==labelnum:
                total=1.0*np.sum((seg>=labelind))
                correct=1.0*np.sum((vote>=labelind)*allcorrect)
                totalpredict=1.0*np.sum((vote>=labelind))
                 
            else:
                total=1.0*np.sum((seg==labelind))
                correct=1.0*np.sum((vote==labelind)*allcorrect)
                totalpredict=1.0*np.sum((vote==labelind))
                
                
            if totalpredict!=0:
                precision[i].append(correct/totalpredict)
            if total!=0:
                recall[i].append(correct/total)
            if totalpredict!=0 or total!=0:
                dice[i].append(2*correct/(total+totalpredict))
    np.save(fold+'_acc.npy',np.array(precision))
    np.save(fold+'_rec.npy',np.array(recall))
    np.save(fold+'_dice.npy',np.array(dice))
print dice[0]
fig = figure()
ax = axes()
#hold(True)
label=['spleen','kidney-r','kidney-l','gallbaldder','esophagus','liver','stomach','aorta','IVC','portalsplenicvein','pancreas','adrenalgland-r','adrenalgland-l','other'
]
print len(label)
assert len(label)==labelnum

gap=10

tick=[]
curtick=int(0.5*gap)
for i in range(labelnum):
    tick.append(curtick)
    curtick=curtick+gap


def bplot(precision,tick,gap,labelnum,fold,saveadd):
    for i in range(labelnum):
        bp = boxplot([precision[i]], positions =[tick[i]], widths = 0.2*gap)
        setBoxColors(bp)



    xlim(0,gap*labelnum)
    ylim(0,1)
    ax.set_xticklabels(label)
    ax.set_xticks(tick)
    ylabel=np.arange(0,1,0.1)
    print ylabel
    ax.set_yticks(ylabel)

    #legend
    #hB, = plot([1,1],'b-')
    #hR, = plot([1,1],'r-')
    #legend((hB, hR),('Apples', 'Oranges'))
    #hB.set_visible(False)
    #hR.set_visible(False)

    savefig(saveadd)
    show()
#bplot(precision,tick,gap,labelnum,fold,fold+'_acc.png')
bplot(dice,tick,gap,labelnum,fold,fold+'_dice.png')    

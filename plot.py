from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import numpy as np
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
fold='norandomrun_1'
resdir="../res/"+fold+"/"
objectNum=25


if os.path.exists(fold+'.npy'):
    precision=np.load(fold+'.npy')
else:
    precision=[]
    recall=[]

    for i in range(labelnum):
        precision.append([])

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
                pre=np.sum((vote>=labelind)*allcorrect)*1.0/np.sum((seg>=labelind))
                precision[i].append(pre)
            else:
                totalnum=np.sum((seg==labelind)) 
                if totalnum!=0:
                    pre=np.sum((vote==labelind)*allcorrect)*1.0/np.sum((seg==labelind))
                    precision[i].append(pre)
                    print pre
    np.save(fold+'.npy',np.array(precision))

#
#print precision[0]
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
print tick
# first boxplot pair


for i in range(labelnum):
    bp = boxplot([precision[i]], positions =[tick[i]], widths = 0.2*gap)
    setBoxColors(bp)



xlim(0,gap*labelnum)
ylim(0,1)
ax.set_xticklabels(label)
ax.set_xticks(tick)


#legend
#hB, = plot([1,1],'b-')
#hR, = plot([1,1],'r-')
#legend((hB, hR),('Apples', 'Oranges'))
#hB.set_visible(False)
#hR.set_visible(False)

savefig(fold+'.png')
show()
    

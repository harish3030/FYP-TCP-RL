#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
def plot_graphs(history,tit,label,episode):

    plt.plot(range(len(history)),history,marker="",linestyle="-")
    plt.title(tit)
    plt.xlabel('Steps')
    plt.ylabel(label)
    output_dir = f"./Plots/Episode-{episode}"
    mkdir_p(output_dir)
    file_name=f'{tit}.png'
    plt.savefig(f"{output_dir}/{file_name}")
    plt.show()


cwnd=[]
tp=[]
rtt=[]
searchfile=open("run.log","r")
for line in searchfile:
    if "\t[$] Congestion Window: " in line: 
        for z in line.split():
             if z.isdigit():
                cwnd.append(int(z))
    if "\t[!] RTT :" in line: 
        for z in line.split():
             if z.isdigit():
                rtt.append(int(z)*1e-6)
    if "\t[!] Throughput: " in line: 
        for z in line.split():
             if z.isdigit():
                 tp.append(int(z)*8)
     
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})
plot_graphs(cwnd,'Congestion Windows','CWND (segments)',1)
plot_graphs(tp,'Throughput over time','Throughput (Mbits/s)',1)
plot_graphs(rtt,'RTT over time','RTT (seconds)',1)
# plot_graphs(rew_history,'Reward sum plot','Accumulated reward',e)
searchfile.close()
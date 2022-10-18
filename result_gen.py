from base_code import *
import os
import pickle
import time
from tqdm import tqdm
#------------------------------------------------------------------------------
LOG_LEVEL = 1
def A_anal(name):
    SAMPLE_RATE = 1_000
    prefix = name[:2] + r'Results\\' + name[2:-30]

    filepath = name
    if LOG_LEVEL > 0:
        start = time.time()
    timeSeries = import_data(filepath,[410.2059233735335,-2.2])
    if LOG_LEVEL > 0:
        print("Data Importing took")
        print(time.time()-start)
        start = time.time()
    detrended = detrend_data(timeSeries)
    if LOG_LEVEL > 0:
        print("Data Detrending took")
        print(time.time()-start)
        start = time.time()
    filtered = filter_data(detrended, 3, [1,15] , 'bandpass', SAMPLE_RATE)
    if LOG_LEVEL > 0:
        print("Data Filtering took")
        print(time.time()-start)
        start = time.time()
    power = power_spectrum(filtered, SAMPLE_RATE)
    if LOG_LEVEL > 0:
        print("Data FFT took")
        print(time.time()-start)
        start = time.time()
    power2 = power_spectrum(filtered, SAMPLE_RATE,True)
    if LOG_LEVEL > 0:
        print("Data FFT2 took")
        print(time.time()-start)
        start = time.time()

    #white_noise = white_noise_checker(timeSeries)
  

    #with open(f'{prefix}_whitenoise_raw.pkl', 'wb') as f:
    #    pickle.dump(white_noise, f)

    #del(white_noise)
    #white_noise2 = white_noise_checker(filtered)
  
    #with open(f'{prefix}_whitenoise_filtered.pkl', 'wb') as f:
    #  pickle.dump(white_noise2, f)

    #del(white_noise2)
    stats = calculate_statistics(filtered)
    if LOG_LEVEL > 0:
        print("Data Statistics took")
        print(time.time()-start)
        start = time.time()
    with open(rf'{prefix}_statistics.pkl', 'wb') as f:
      pickle.dump(stats, f)

    if LOG_LEVEL > 0:
        print("Stats dumping took")
        print(time.time()-start)
        start = time.time()

    plot_time_series(timeSeries)
    plt.savefig(rf'{prefix}_timeseries_raw.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting time took")
        print(time.time()-start)
        start = time.time()

    plot_time_series(timeSeries,lim=[1000,2000])
    plt.savefig(rf'{prefix}_time_shortened.png')
    plt.close()
    
    if LOG_LEVEL > 0:
        print("Plotting time shor took")
        print(time.time()-start)
        start = time.time()

    plot_time_series(detrended)
    plt.savefig(rf'{prefix}_timeseries_detrended.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting detrended took")
        print(time.time()-start)
        start = time.time()

    plot_time_series(detrended,lim=[1000,2000])
    plt.savefig(rf'{prefix}_detrended_shortened.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting detrended short took")
        print(time.time()-start)
        start = time.time()

    plot_time_series(filtered)
    plt.savefig(rf'{prefix}_timeseries_filtered.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting filtered took")
        print(time.time()-start)
        start = time.time()

    plot_time_series(filtered,lim=[1000,2000])
    plt.savefig(rf'{prefix}_filtered_shortened.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting filtered short took")
        print(time.time()-start)
        start = time.time()

    plot_power_spectrum(power,logs = True,lims=[0,50])
    plt.savefig(rf'{prefix}_power.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting power took")
        print(time.time()-start)
        start = time.time()
    
    plot_power_spectrum(power2,logs = True,lims=[0,50])
    plt.savefig(rf'{prefix}_power_disp.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Plotting power2 took")
        print(time.time()-start)
        start = time.time()

    cross_correlation(filtered, ['1Z','2Z','3Z','4Z','5Z','6Z','7Z','8Z'],plot=True,norm=True,include_sample_rate=SAMPLE_RATE)
    plt.savefig(rf'{prefix}_correlation_Z.png')
    plt.close()

    cross_correlation(filtered,['3X','4X','5X'],plot=True,norm=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(rf'{prefix}_correlation_X.png')
    plt.close()

    cross_correlation(filtered,['3Y','4Y','5Y'],plot=True,norm=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(rf'{prefix}_correlation_Y.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Correlations took")
        print(time.time()-start)
        start = time.time()

    coherence(filtered, ['1Z','2Z','3Z','4Z','5Z','6Z','7Z','8Z'],plot=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(rf'{prefix}_coherence_Z.png')
    plt.close()

    coherence(filtered,['3X','4X','5X'],plot=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(rf'{prefix}_coherence_X.png')
    plt.close()

    coherence(filtered,['3Y','4Y','5Y'],plot=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(rf'{prefix}_coherence_Y.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Coherence took")
        print(time.time()-start)
        start = time.time()

    plot_kde(filtered,['1Z','2Z','3Z','4Z','5Z','6Z','7Z','8Z','3X','4X','5X','3Y','4Y','5Y'])
    plt.savefig(rf'{prefix}_kdehist.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("KDE took")
        print(time.time()-start)
        start = time.time()

    plot_power_spectrum_density(filtered,SAMPLE_RATE, ['1Z','2Z','3Z','4Z','5Z','6Z','7Z','8Z','3X','4X','5X','3Y','4Y','5Y'],lims=[0,50])
    plt.savefig(rf'{prefix}_psd.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("PSD took")
        print(time.time()-start)
        start = time.time()
    
    data = turbulent_signal_checker(power,[1,15])
    y = []
    x = []
    for k in data.keys():
        y.append(data[k])
        x.append(k)

    plt.plot(np.abs(y),'x')
    plt.xticks(range(len(y)),x)
    plt.plot(np.zeros(len(y))+7/3)
    plt.savefig(rf'{prefix}_loglogdecay.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Line took")
        print(time.time()-start)
        start = time.time()
    
    data = turbulent_signal_checker(power2,[1,15])
    y = []
    x = []
    for k in data.keys():
        y.append(data[k])
        x.append(k)

    plt.plot(np.abs(y),'x')
    plt.xticks(range(len(y)),x)
    plt.plot(np.zeros(len(y))+7/3)
    plt.savefig(rf'{prefix}_loglogdecay_disp.png')
    plt.close()

    if LOG_LEVEL > 0:
        print("Line2 took")
        print(time.time()-start)
        
    
    return 1

def EA_anal(name):
    SAMPLE_RATE = 1_000
    prefix = name[:3] + r'Results\\' + name[3:-30] + 'external'
    filepath = name
    timeSeries = import_data(filepath)
    detrended = detrend_data(timeSeries)
    filtered = filter_data(detrended, 3, [1,15] , 'bandpass', SAMPLE_RATE)
    power = power_spectrum(filtered, SAMPLE_RATE)
    power2 = power_spectrum(filtered, SAMPLE_RATE,True)
    #white_noise = white_noise_checker(timeSeries)
  

    #with open(f'{prefix}_whitenoise_raw.pkl', 'wb') as f:
     #   pickle.dump(white_noise, f)

    #del(white_noise)
    #white_noise2 = white_noise_checker(filtered)
  
    #with open(f'{prefix}_whitenoise_filtered.pkl', 'wb') as f:
     # pickle.dump(white_noise2, f)

    #del(white_noise2)
    stats = calculate_statistics(filtered)
    with open(f'{prefix}_statistics.pkl', 'wb') as f:
        pickle.dump(stats, f)

    plot_time_series(timeSeries)
    plt.savefig(f"{prefix}_timeseries_raw.png")
    plt.close()

    plot_time_series(timeSeries,lim=[1000,2000])
    plt.savefig(f"{prefix}_time_shortened.png")
    plt.close()

    plot_time_series(detrended)
    plt.savefig(f"{prefix}_timeseries_detrended.png")
    plt.close()

    plot_time_series(detrended,lim=[1000,2000])
    plt.savefig(f"{prefix}_detrended_shortened.png")
    plt.close()

    plot_time_series(filtered)
    plt.savefig(f"{prefix}_timeseries_filtered.png")
    plt.close()

    plot_time_series(filtered,lim=[1000,2000])
    plt.savefig(f"{prefix}_filtered_shortened.png")
    plt.close()

    plot_power_spectrum(power,logs = True,lims=[0,50])
    plt.savefig(f"{prefix}_power.png")
    plt.close()

    cross_correlation(filtered,['DS Tank','Frame X','Frame Y','Frame Z'],plot=True,norm=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(f"{prefix}_Frame_correlation.png")
    plt.close()

    cross_correlation(filtered,['DS Tank','Pipe X','Pipe Y','Pipe Z'],plot=True,norm=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(f"{prefix}_Pipe_correlation.png")
    plt.close()

    coherence(filtered, ['DS Tank','Frame X','Frame Y','Frame Z'],plot=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(f"{prefix}_coherence_frame.png")
    plt.close()

    coherence(filtered,['DS Tank','Pipe X','Pipe Y','Pipe Z'],plot=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(f"{prefix}_coherence_pipe.png")
    plt.close()

    plot_kde(filtered,['DS Tank','US tank','Frame X','Frame Y','Frame Z','Pipe X','Pipe Y','Pipe Z'])
    plt.savefig(f"{prefix}_kdehist.png")
    plt.close()

    plot_power_spectrum_density(filtered,SAMPLE_RATE, ['DS Tank','Frame X','Frame Y','Frame Z'],lims=[0,50])
    plt.savefig(f"{prefix}_psd_frame.png")
    plt.close()

    plot_power_spectrum_density(filtered,SAMPLE_RATE, ['DS Tank','Pipe X','Pipe Y','Pipe Z'],lims=[0,50])
    plt.savefig(f"{prefix}_psd_pipe.png")
    plt.close()

    data = turbulent_signal_checker(power,[1,15])
    y = []
    x = []
    for k in data.keys():
        y.append(data[k])
        x.append(k)

    plt.plot(np.abs(y),'x')
    plt.xticks(range(len(y)),x)
    plt.plot(np.zeros(len(y))+7/3)
    plt.savefig(rf'{prefix}_loglogdecay.png')
    plt.close()
    
    data = turbulent_signal_checker(power2,[1,15])
    y = []
    x = []
    for k in data.keys():
        y.append(data[k])
        x.append(k)

    plt.plot(np.abs(y),'x')
    plt.xticks(range(len(y)),x)
    plt.plot(np.zeros(len(y))+7/3)
    plt.savefig(rf'{prefix}_loglogdecay_disp.png')
    plt.close()
      

    return 1

def FBG_anal(name):
    SAMPLE_RATE = 1_000
    prefix = name[:4] + r'Results\\' + name[4:-20] 
    filepath = name
    timeSeries = import_fbg_data(filepath)
    timeSeries.pop('Time')
    timeSeries.pop(15)
    timeSeries.pop(16)
    timeSeries.pop(17)
    timeSeries.pop(18)
    timeSeries.pop(19)
    timeSeries.pop(20)
    timeSeries.pop(21)
    timeSeries.pop(22)
    timeSeries.pop(23)
    timeSeries.pop(24)

    timeSeries.pop('Number of Sensors')
    
    detrended = detrend_data(timeSeries)
    filtered = filter_data(detrended, 3, [1,15] , 'bandpass', SAMPLE_RATE)
    power = power_spectrum(filtered, SAMPLE_RATE)
    stats = calculate_statistics(filtered)
    
    with open(rf'{prefix}_statistics.pkl', 'wb') as f:
      pickle.dump(stats, f)

    plot_time_series(timeSeries)
    plt.savefig(rf'{prefix}_timeseries_raw.png')
    plt.close()

    plot_time_series(timeSeries,lim=[1000,2000])
    plt.savefig(rf'{prefix}_time_shortened.png')
    plt.close()

    plot_time_series(detrended)
    plt.savefig(rf'{prefix}_timeseries_detrended.png')
    plt.close()

    plot_time_series(detrended,lim=[1000,2000])
    plt.savefig(rf'{prefix}_detrended_shortened.png')
    plt.close()

    plot_time_series(filtered)
    plt.savefig(rf'{prefix}_timeseries_filtered.png')
    plt.close()

    plot_time_series(filtered,lim=[1000,2000])
    plt.savefig(rf'{prefix}_filtered_shortened.png')
    plt.close()

    plot_power_spectrum(power,logs = True,lims=[0,50])
    plt.savefig(rf'{prefix}_power.png')
    plt.close()
    
    
    cross_correlation(filtered, ['0','1','2','3','4','5','6','7','8','9'],plot=True,norm=True,include_sample_rate=SAMPLE_RATE)
    plt.savefig(f"{prefix}_correlation.png")
    plt.close()
    
    coherence(filtered, ['0','1','2','3','4','5','6','7','8','9'],plot=True,include_sample_rate = SAMPLE_RATE)
    plt.savefig(f"{prefix}_coherence.png")
    plt.close()
    
    plot_kde(filtered,['0','1','2','3','4','5','6','7','8','9'])
    plt.savefig(f"{prefix}_kdehist.png")
    plt.close()

    plot_power_spectrum_density(filtered,SAMPLE_RATE, ['0','1','2','3','4','5','6','7','8','9'],lims=[0,50])
    plt.savefig(f"{prefix}_psd.png")
    plt.close()

    data = turbulent_signal_checker(power,[1,15])
    y = []
    x = []
    for k in data.keys():
        y.append(data[k])
        x.append(k)

    plt.plot(np.abs(y),'x')
    plt.xticks(range(len(y)),x)
    plt.plot(np.zeros(len(y))+7/3)
    plt.savefig(rf'{prefix}_loglogdecay.png')
    plt.close()
    return 1


#----------------------------------------------------------------------------


print("Analysis Running")
directories_in_curdir = list(filter(os.path.isdir, os.listdir(os.getcwd())))
dirs = []
for dirr in directories_in_curdir:
    if dirr == 'A' or dirr == 'EA' or dirr == 'FBG':
        dirs.append(dirr)
        
for d in tqdm(dirs):
    if d == 'A':
        
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        #files = os.listdir(d)
        print(files)
        if not os.path.isdir(os.path.join(d, "Results")):
            os.mkdir(os.path.join(d, "Results"))
        for f in tqdm(files):
            paths = os.path.join(d, f)
            A_anal(paths)
        print("Acceleremeters Finished")

        

        
        
    elif d == 'EA':
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        if not os.path.isdir(os.path.join(d, "Results")):
            os.mkdir(os.path.join(d, "Results"))
        for f in tqdm(files):
            paths = os.path.join(d, f)
            EA_anal(paths)
        print("External Acceleremeters Finished")

        
        
    else:
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        if not os.path.isdir(os.path.join(d, "Results")):
            os.mkdir(os.path.join(d, "Results"))
        for f in tqdm(files):
	    
            paths = os.path.join(d, f)
            print(paths)
            FBG_anal(paths)
        print("FBG Finished")

        


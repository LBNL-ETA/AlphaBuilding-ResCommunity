
import numpy as np

def plotStep(x, y, ax, style = None, **kwargs):
    ## Pad the first value
    y = np.pad(y, (1, 0), mode = 'edge')
    if style is None:
        l, = ax.step(x, y)
    elif kwargs is None:
        l, = ax.step(x, y, style)
    else:
        l, = ax.step(x, y, style, **kwargs)
    return l
    
def plot_prediction(df, predictions, ax, state_name = "T_ctrl", start_day = None, end_day = None, hours = [1, 6], step_per_hour = 4):

    ax.plot(df.index.to_pydatetime(), df[state_name], 'k')
    for hour in hours:
        idx = df.index[hour*step_per_hour:].to_pydatetime()
        ax.plot(idx[:len(predictions[hour])], predictions[hour][:len(idx)], label = "{}-hour ahead".format(hour))
    ax.legend()

    if start_day is None:
        start_day = df.index[0]
    if end_day is None:
        end_day = df.index[-1]

    ax.set_xlim(start_day, end_day)
    return ax

def plotDensity(states, ax, bins, deadband, T = 24):
    p_list = []
    for row in states:
        p, _ = np.histogram(row, bins = bins)#, density = True)
        p_list.append(p)
    m = np.array(p_list)

    for item in deadband:
        ax.plot((0, T), (item, item), 'g--')
    #ax.plot((0, 24), (bins[1], bins[1]), color = 'red')
    #ax.plot((0, 24), (bins[-2], bins[-2]), color = 'red')
    Delta = (bins[2]-bins[1])*2
    ax.imshow(m.T, aspect = 'auto', cmap='Greys', extent=[0, T, bins[1]-Delta/2, bins[-2]+Delta/2], origin = 'lower')
    #ax.set_ylabel("Zone Temp. (C)")
    ax.set_ylim(bins[1]-Delta/2, bins[-2]+Delta/2)
    return m

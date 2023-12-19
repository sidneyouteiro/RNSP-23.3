import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

import wisardpkg as wp

from encoders import process_input, Morph
from datagen import add_noise, sample_mental_image

def plot_decision_boundary(model, encoder, axis=None, x_range=[0.0, 1.0], y_range=[0.0, 1.0], step=0.005, title="Decision Boundary"):
    xmin, xmax = x_range
    ymin, ymax = y_range
    
    x, y = np.arange(xmin, xmax, step), np.arange(ymax, ymin, -step)
    X, Y = np.meshgrid(x, y)

    # predict = lambda x, y: int(model.classify(process_input([[x, y]], encoder))[0])
    predict = lambda x, y: int(
        model.classify(
            np.atleast_2d(
                process_input([[x, y]], encoder=encoder)).tolist())[0])
    
    axis = plt.subplot() if axis is None else axis
    
    axis.set_title(title)
    return axis.contourf(X, Y, np.vectorize(predict)(X, Y))


def plot_imagined_points(model, encoder, kargs, axis=None, seed=None, alpha=0.5, title="Imagined Points"):
    mental_images = model.getMentalImages()
    
    # Normalize mental images
    for k in mental_images.keys():
        # TODO: This normalization may not be right!
        mental_images[k] = np.asarray(mental_images[k])/np.amax(mental_images[k])
        
    imagined_points = np.concatenate([
        encoder.decode(sample_mental_image(mental_images[k], kargs[k][1], seed=seed))
        for k in mental_images.keys()
    ])
    
    imagined_y = np.concatenate([
        np.repeat(k, kargs[k][1])
        for k in mental_images.keys()
    ]).astype(int)
    
    axis = plt.subplot() if axis is None else axis
    axis.set_title(title)
    axis.scatter(imagined_points[:, 0], imagined_points[:, 1], c=imagined_y, s=15, alpha=alpha)


def accuracy(y_pred, y_target):
    y_pred = np.asarray(y_pred)
    y_target = np.asarray(y_target)

    return np.where(y_pred == y_target, 1, 0).sum()/len(y_target)


def extrema(x):
    x = np.asarray(x)
    
    return (np.min(x), np.max(x))


def square_layout(nplots):
    nrows = 1
    ncols = 1
    
    while nplots > nrows*ncols:
        if nrows < ncols:
            nrows += 1
        else:
            ncols +=1
            
    return nrows, ncols


def classifier_testbed(X, y, tuple_size, noise, encoders, seed, bleaching=True, layout='square', subplot_size=(6, 6), mental_images=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    
    # Add noise to training set
    y_train = add_noise(y_train, noise, seed)
    
    models = {k: wp.Wisard(tuple_size, bleachingActivated=bleaching) for k in encoders.keys()}
    
    # Train models and assess accuracy
    for k in encoders.keys():
        try:
            encoder_train, encoder_test = encoders[k]
        except TypeError:
            encoder_train = encoder_test = encoders[k]
        except ValueError:
            encoder_train = encoder_test = encoders[k][0]
        
        models[k].train(
#             encoder_train.encode(X_train),
            # process_input(X_train, encoder=encoders[k]),
            process_input(X_train, encoder=encoder_train),
            y_train.astype(str).tolist()
        )
        
        print(f"{k}:",
            accuracy(
                models[k].classify(
#                     encoder_test.encode(X_test),
                    # process_input(X_test, encoder=encoders[k])
                    process_input(X_test, encoder=encoder_test)
                ),
                y_test.astype(str)
            )
        )
    
    nplots = len(encoders)*(int(mental_images) + 1) + 1
    if layout == 'square':
        nrows, ncols = square_layout(nplots)
    elif layout == 'column':
        nrows = nplots
        ncols = 1
    else:
        nrows = 1
        ncols = nplots
        
    gs = GridSpec(nrows, ncols)
    fig = plt.figure(figsize=(subplot_size[0]*ncols, subplot_size[1]*nrows))
    
    plot_index = 0
    ax = fig.add_subplot(gs[plot_index])
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10)
    ax.set_title('Training Data')
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_range = extrema(X[:, 0])
    y_range = extrema(X[:, 1])
    plot_index += 1
    
    if mental_images:
        # Need to fix the encoding and morphing part
        for k in encoders.keys():
            try:
                encoder_train, encoder_test = encoders[k]
            except TypeError:
                encoder_train = encoder_test = encoders[k]
            except ValueError:
                encoder_train = encoder_test = encoders[k][0]
                
            ax = fig.add_subplot(gs[plot_index])
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            plot_imagined_points(
                models[k],
                encoder_train, # Any encoder should do for decoding
                {
                    '1':  (sum(np.where(y_train == 1, 1, 0)), sum(np.where(y_train == 1, 1, 0))),
                    '-1': (sum(np.where(y_train == -1, 1, 0)), sum(np.where(y_train == -1, 1, 0)))
                },
                axis=ax,
                seed=seed,
                title=f"Mental image ({k})"
            )
            plot_index += 1
    
    for i, k in enumerate(encoders.keys()):
        try:
            encoder_train, encoder_test = encoders[k]
        except TypeError:
            encoder_train = encoder_test = encoders[k]
        except ValueError:
            encoder_train = encoder_test = encoders[k][0]
            
        ax = fig.add_subplot(gs[plot_index+i])
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        plot_decision_boundary(
            models[k],
            encoder_test,
            ax,
            x_range,
            y_range,
            title=f"Decision boundary ({k})"
        )
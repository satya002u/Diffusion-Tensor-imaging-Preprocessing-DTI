from deepexplain.tensorflow import DeepExplain
from keras import backend as K
import numpy as np
from keras.models import Sequential, Model

def get_contr_score_deeplift(model, xs, ys):
  with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    out = model.layers[-2].output
    target_tensor = fModel(input_tensor)  
  
    attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)

    return attributions_dl
    # plt.imshow(attributions_dl)
    # plt.show()
    
def get_contr_score_grad_input(model, xs, ys):
  with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    out = model.layers[-2].output
    target_tensor = fModel(input_tensor)     
    attributions_dl = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)  
    return attributions_dl
    #
def get_contr_score_saliency(model, xs, ys):
  with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    out = model.layers[-2].output
    target_tensor = fModel(input_tensor)     
    attributions_dl   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys) 
    return attributions_dl

def get_contr_score_intgrad(model, xs, ys):
  with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    out = model.layers[-2].output
    target_tensor = fModel(input_tensor)     
    attributions_dl   = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys) 
    return attributions_dl

def get_contr_score_elrp(model, xs, ys):
  with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    out = model.layers[-2].output
    target_tensor = fModel(input_tensor)     
    attributions_dl   = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys) 
    return attributions_dl
    #
def get_contr_score_occlusion(model, xs, ys):
  with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    out = model.layers[-2].output
    target_tensor = fModel(input_tensor)     
    attributions_dl   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys) 
    return attributions_dl
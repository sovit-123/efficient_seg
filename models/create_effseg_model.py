from models import *

def create_effseg4_16s(num_classes, aux):
    model = effseg4_16s.EffSegModel(num_classes, aux=aux)
    return model

def create_effseg0_16s(num_classes, aux):
    model = effseg0_16s.EffSegModel(num_classes, aux=aux)
    return model

def create_effseg4_8s(num_classes, aux):
    model = effseg4_8s.EffSegModel(num_classes, aux=aux)
    return model

create_model = {
    'effseg0_16s': create_effseg0_16s, 
    'effseg4_16s': create_effseg4_16s,
    'effseg4_8s': create_effseg4_8s
}
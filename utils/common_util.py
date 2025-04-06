import os
import torch

def forward_model(model, x_rgb, x_d, x_ir, modality):
    if modality == 'RGB':
        return model(x_rgb)
    if modality == 'D':
        return model(x_d)
    if modality == 'RGBD':
        return model(x_rgb, x_d)
    if modality == 'RGBIR':
        return model(x_rgb, x_ir)
    if modality == 'RGBDIR':
        return model(x_rgb, x_d, x_ir)

def forward_model_with_domain(model, x_rgb, x_d, x_ir, modality, domain):
  if modality == 'RGB':
      return model(x_rgb, domain)
  if modality == 'D':
      return model(x_d, domain)
  if modality == 'RGBD':
      return model(x_rgb, x_d, domain)
  if modality == 'RGBIR':
      return model(x_rgb, x_ir, domain)
  if modality == 'RGBDIR':
      return model(x_rgb, x_d, x_ir, domain)

def forward_clip_model(model, x_rgb, x_d, x_ir, modality, mode='train', missing=[]):
    if mode == 'train':
        if modality == 'RGB':
            return model(x_rgb)
        if modality == 'D':
            return model(x_d)
        if modality == 'IR':
            return model(x_ir)
        if modality == 'RGBD':
            return model(x_rgb, x_d)
        if modality == 'DIR':
            return model(x_d, x_ir)
        if modality == 'RGBIR':
            return model(x_rgb, x_ir)
        if modality == 'RGBDIR':
            return model(x_rgb, x_d, x_ir, missing)
    elif mode == 'test':
        if modality == 'RGB':
            return model.forward_test(x_rgb)
        if modality == 'D':
            return model.forward_test(x_d)
        if modality == 'IR':
            return model.forward_test(x_ir)
        if modality == 'RGBD':
            return model.forward_test(x_rgb, x_d)
        if modality == 'DIR':
            return model.forward_test(x_d, x_ir)
        if modality == 'RGBIR':
            return model.forward_test(x_rgb, x_ir)
        if modality == 'RGBDIR':
            return model.forward_test(x_rgb, x_d, x_ir, missing)

def mkdir_if_missing(dirname):
  """Create dirname if it is missing."""
  if not os.path.exists(dirname):
      try:
          os.makedirs(dirname)
      except OSError as e:
          if e.errno != errno.EEXIST:
              raise

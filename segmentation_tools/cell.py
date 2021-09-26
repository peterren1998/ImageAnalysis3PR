import numpy as np


class Cellpose_Segmentation:
    """"""
    def __init__(self, _im, data_type='DAPI', 
                 save_filename=None, verbose=True):
        """"""
        # inherit from superclass
        super().__init__()
    
        # save images
        self.raw_im = _im
        self.data_type = data_type
        self.allowed_types = ['DAPI', 'polyT']
        if self.data_type not in self.allowed_types:
            raise ValueError(f"input datatype {self.data_type} not in {self.allowed_types}")
        # save 
        self.save_filename = save_filename
        self.verbose = verbose

    def run(self):
        """"""
        _lys, _sel_ids = Cellpose_Segmentation.pick_Z_stacks(self.raw_im)
        
        _mask = Cellpose_Segmentation.run_segmentation(_lys)
        
        _clean_mask = Cellpose_Segmentation.merge_3d_masks(_mask)
        
        _z = Cellpose_Segmentation.convert_layer_list_to_um(_sel_ids)
        _full_mask = Cellpose_Segmentation.interploate_z_masks(_clean_mask, _z)
        
        return _full_mask
        
    @staticmethod
    def pick_Z_stacks(im:np.ndarray, 
                      num_layer_project:int=5,
                      num_layer_overlap:int=1,
                      projection_method:'function'=np.mean,
                      verbose=True,
                      ):
        
        _im = im.copy()
        # projection on z
        _sel_layers = []
        for _i, _ly in enumerate(_im):
            if _i < num_layer_project-1:
                continue
            if len(_sel_layers) == 0 or min(_sel_layers[-1][-1*num_layer_overlap-1:]) + num_layer_project <= _i:
                _sel_layers.append(np.arange(_i-num_layer_project+1, _i+1))
                
        # generate max projections
        _max_proj_layers = np.array([projection_method(_im[np.array(_lys)],axis=0) for _lys in _sel_layers])
        
        if verbose:
            print(f"- {len(_max_proj_layers)} layers selected with {projection_method} projection.")
        return _max_proj_layers, _sel_layers
    
    @staticmethod
    def run_segmentation(_projected_im, 
                          model_type='nuclei', 
                          use_gpu=True, 
                          diameter=60, min_size=10, 
                          cellprob_threshold=0.5, stitch_threshold=0.2,
                          flow_threshold=1.0,
                          verbose=True,
                          ):
        from cellpose import models
        # segmentation
        seg_model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        masks, _, _, _ = seg_model.eval(
            np.array([_projected_im,_projected_im]),
            z_axis=1,
            channel_axis=0,
            diameter=diameter, 
            channels=[0,0], 
            min_size=min_size,
            cellprob_threshold=cellprob_threshold, # -6 to 6, positively correlate with number of masks
            stitch_threshold=stitch_threshold,
            flow_threshold=flow_threshold,
            do_3D=False)
        # clear ram
        del(seg_model)
        
        return masks
    
    @staticmethod
    def merge_3d_masks(masks, overlap_th=0.9, verbose=True):
        
        import time
        # copy masks
        _masks = np.array(masks).copy()
        all_mask_ids = np.unique(_masks)
        all_mask_ids = all_mask_ids[all_mask_ids>0]

        xy_projections = [(_masks==_i).any(0) for _i in all_mask_ids]

        kept_masks = np.zeros(np.shape(_masks), dtype=np.uint16)
        kept_ids = []

        # intialize
        if verbose:
            print(f"- start merging 3d masks")
            _start_time = time.time()
        unprocessed_ids = list(all_mask_ids)

        while len(unprocessed_ids) > 0:
            # default: kept this cell
            _kept_flag = True
            # extract i
            _i = unprocessed_ids.pop(0)
            _i_msk = xy_projections[list(all_mask_ids).index(_i)]

            # calculate j percentage to see whether merge this into _j
            for _j in unprocessed_ids:
                # extract j
                _j_msk = xy_projections[list(all_mask_ids).index(_j)]

                # compare these two masks
                _i_percent = np.sum(_i_msk*_j_msk) / np.sum(_i_msk)
                _j_percent = np.sum(_i_msk*_j_msk) / np.sum(_j_msk)
                if _i_percent > 0 or _j_percent > 0:
                    if verbose:
                        print(f"-- overlap found for cell:{_i} to {_j}", _i_percent, _j_percent)

                # remove i, merge into j
                if _i_percent > overlap_th:
                    _kept_flag = False
                    # update mask, i already removed by continue
                    _masks[_masks==_i] = _j
                    xy_projections[list(all_mask_ids).index(_j)] = (_masks==_j).any(0)
                    if verbose:
                        print(f"--- skip {_i}")
                    break
                    
                # remove j, merge into i
                elif _j_percent > overlap_th:
                    _kept_flag = False
                    # remove j
                    unprocessed_ids.pop(unprocessed_ids.index(_j))
                    # update mask
                    _masks[_masks==_j] = _i
                    xy_projections[list(all_mask_ids).index(_i)] = (_masks==_i).any(0)
                    # redo i
                    unprocessed_ids = [_i] + unprocessed_ids
                    if verbose:
                        print(f"--- redo {_i}")
                    break

            # save this mask if there's no overlap
            if _kept_flag:
                kept_masks[_masks==_i] = np.max(np.unique(kept_masks))+1
                kept_ids.append(_i)
        if verbose:
            print(f"- {np.max(kept_masks)} labels kept.")
            print(f"- finish in {time.time()-_start_time:.2f}s. ")
            
        return kept_masks
    
    @staticmethod
    def convert_layer_list_to_um(layer_lists:list, 
                                 step_sizes:float=0.2, 
                                 select_method:'function'=np.median):
        return step_sizes * np.array([select_method(_lys) for _lys in layer_lists])
    
    @staticmethod
    def interploate_z_masks(z_masks, 
                            z_coords, 
                            target_z_coords=np.round(np.arange(0,12,0.2),2),
                            mode='nearest',
                            verbose=True,
                            ):

        # target z
        _final_mask = []
        _final_coords = np.round(target_z_coords, 3)
        for _fz in _final_coords:
            if _fz in z_coords:
                _final_mask.append(z_masks[np.where(z_coords==_fz)[0][0]])
            else:
                if mode == 'nearest':
                    _final_mask.append(z_masks[np.argmin(np.abs(z_coords-_fz))])
                    continue
                # find nearest neighbors
                if np.sum(z_coords > _fz) > 0:
                    _upper_z = np.min(z_coords[z_coords > _fz])
                else:
                    _upper_z = np.max(z_coords)
                if np.sum(z_coords < _fz) > 0:
                    _lower_z = np.max(z_coords[z_coords < _fz])
                else:
                    _lower_z = np.min(z_coords)

                if _upper_z == _lower_z:
                    # copy the closest mask to extrapolate
                    _final_mask.append(z_masks[np.where(z_coords==_upper_z)[0][0]])
                else:
                    # interploate
                    _upper_mask = z_masks[np.where(z_coords==_upper_z)[0][0]].astype(np.float32)
                    _lower_mask = z_masks[np.where(z_coords==_lower_z)[0][0]].astype(np.float32)
                    _inter_mask = (_upper_z-_fz)/(_upper_z-_lower_z) * _lower_mask 
                    
        if verbose:
            print(f"- reconstruct {len(_final_mask)} layers")
        
        return np.array(_final_mask)
    
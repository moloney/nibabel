"""
DcmMeta header extension and NiftiWrapper for working with extended Niftis.
"""
import sys
import json, warnings
from copy import deepcopy
import numpy as np
from ..nifti1 import Nifti1Extension
from ..spatialimages import HeaderDataError

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

dcm_meta_ecode = 0

_meta_version = 0.7

_req_base_keys_map= {0.5 : set(('dcmmeta_affine', 
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'global',
                                )
                               ),
                     0.6 : set(('dcmmeta_affine', 
                                'dcmmeta_reorient_transform',
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'global',
                                )
                               ),
                     0.7 : set(('dcmmeta_affine',
                                'dcmmeta_reorient_transform',
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'const',
                                'per_slice',
                               )
                              )
                    }
'''Minimum required keys in the base dictionary to be considered valid'''

def is_constant(sequence, period=None):
    '''Returns true if all elements in (each period of) the sequence are equal. 
    
    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.
    
    period : int
        If not None then each subsequence of that length is checked. 
    '''
    if period is None:
        return all(val == sequence[0] for val in sequence)
    else:
        if period <= 1:
            raise ValueError('The period must be greater than one')
        seq_len = len(sequence)
        if seq_len % period != 0:
            raise ValueError('The sequence length is not evenly divisible by '
                             'the period length.')
                             
        for period_idx in range(seq_len / period):
            start_idx = period_idx * period
            end_idx = start_idx + period
            if not all(val == sequence[start_idx] 
                       for val in sequence[start_idx:end_idx]):
                return False
    
    return True
    
def is_repeating(sequence, period):
    '''Returns true if the elements in the sequence repeat with the given 
    period.

    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.
        
    period : int
        The period over which the elements should repeat.    
    '''
    seq_len = len(sequence)
    if period <= 1 or period >= seq_len:
        raise ValueError('The period must be greater than one and less than '
                         'the length of the sequence')
    if seq_len % period != 0:
        raise ValueError('The sequence length is not evenly divisible by the '
                         'period length.')
                         
    for period_idx in range(1, seq_len / period):
        start_idx = period_idx * period
        end_idx = start_idx + period
        if sequence[start_idx:end_idx] != sequence[:period]:
            return False
    
    return True

class InvalidExtensionError(Exception):
    def __init__(self, msg):
        '''Exception denoting than a DcmMetaExtension is invalid.'''
        self.msg = msg
    
    def __str__(self):
        return 'The extension is not valid: %s' % self.msg
        

class DcmMeta(OrderedDict):
    '''Ordered mapping for storing a summary of the meta data from the source 
    DICOM files used to create a N-d volume.
    '''
    
    @property
    def affine(self):
        '''The affine associated with the meta data. If this differs from the 
        image affine, the per-slice meta data will not be used. '''
        return np.array(self['dcmmeta_affine'])
        
    @affine.setter
    def affine(self, value):
        if value.shape != (4, 4):
            raise ValueError("Invalid shape for affine")
        self['dcmmeta_affine'] = value.tolist()
        
    @property
    def slice_dim(self):
        '''The index of the slice dimension associated with the per-slice 
        meta data.'''
        return self['dcmmeta_slice_dim']
    
    @slice_dim.setter
    def slice_dim(self, value):
        if not value is None and not (0 <= value < 3):
            raise ValueError("The slice dimension must be in the range [0,3)")
        self['dcmmeta_slice_dim'] = value
    
    @property
    def shape(self):
        '''The shape of the data associated with the meta data. Defines the 
        number of values for the meta data classifications.'''
        return tuple(self['dcmmeta_shape'])
    
    @shape.setter
    def shape(self, value):
        if len(value) < 2:
            raise ValueError("There must be at least two dimensions")
        if len(value) == 2:
            value = value + (1,)
        self._content['dcmmeta_shape'][:] = value
    
    @property
    def reorient_transform(self):
        '''The transformation due to reorientation of the data array. Can be 
        used to update directional DICOM meta data (after converting to RAS if 
        needed) into the same space as the affine.'''
        if self._content['dcmmeta_reorient_transform'] is None:
            return None
        return np.array(self._content['dcmmeta_reorient_transform'])
        
    @reorient_transform.setter
    def reorient_transform(self, value):
        if not value is None and value.shape != (4, 4):
            raise ValueError("The reorient_transform must be none or (4,4) "
            "array")
        if value is None:
            self._content['dcmmeta_reorient_transform'] = None
        else:
            self._content['dcmmeta_reorient_transform'] = value.tolist()
            
    @property
    def version(self):
        '''The version of the meta data extension.'''
        return self._content['dcmmeta_version']
        
    @version.setter
    def version(self, value):
        '''Set the version of the meta data extension.'''
        self._content['dcmmeta_version'] = value
        
    @property
    def slice_normal(self):
        '''The slice normal associated with the per-slice meta data.'''
        slice_dim = self.slice_dim
        if slice_dim is None:
            return None
        return np.array(self.affine[slice_dim][:3])
    
    @property
    def n_slices(self):
        '''The number of slices associated with the per-slice meta data.'''
        slice_dim = self.slice_dim
        if slice_dim is None:
            return None
        return self.shape[slice_dim]
        
    def get_valid_classes(self):
        '''Return the meta data classifications that are valid for this 
        extension.
        
        Returns
        -------
        valid_classes : tuple
            The classifications that are valid for this extension (based on its 
            shape).
        
        '''
        shape = self.shape
        n_dims = len(shape)
        result = ['const', 'per_slice']
        if n_dims > 3:
            for dim_idx in xrange(3, n_dims):
                result.append('per_sample_%d' % dim_idx)
        return tuple(result)
        
    per_sample_re = re.compile('per_sample_([0-9]+)')
    
    def get_per_sample_dim(classification):
        if not classification in self.get_valid_classes():
            raise ValueError("Invalid classification: %s" % classification)
        
        match = re.match(self.per_sample_re, classification)
        if not match:
            raise ValueError("Classification is not per_sample")
        return int(match.groups()[0])
            
    def get_n_vals(self, classification):
        '''Get the number of meta data values for all meta data of the provided 
        classification.
        
        Parameters
        ----------
        classification : tuple
            The meta data classification.
            
        Returns
        -------
        n_vals : int
            The number of values for any meta data of the provided 
            `classification`.
        '''
        if not classification in self.get_valid_classes():
            raise ValueError("Invalid classification: %s" % classification)
        
        if classification == 'const':
            n_vals = 1
        elif classification == 'per_slice':
            n_vals = self.n_slices
            for dim_size in self.shape[3:]:
                n_vals *= dim_size
        else:
            dim = self.get_per_sample_dim(classification)
            n_vals = self.shape[dim]
            for dim_size in self.shape[dim+1:]:
                n_vals *= dim_size
        return n_vals
    
    def check_valid(self):
        '''Check if the extension is valid.
        
        Raises
        ------
        InvalidExtensionError 
            The extension is missing required meta data or classifications, or
            some element(s) have the wrong number of values for their 
            classification.
        '''
        #For now we just fail out with older versions of the meta data
        if self.version != _meta_version:
            raise InvalidExtensionError("Meta data version is out of date, "
                                        "you may be able to convert to the "
                                        "current version.")
                                        
        #Check for the required base keys in the json data
        if not _req_base_keys_map[self.version] <= set(self._content):
            raise InvalidExtensionError('Missing one or more required keys')
            
        #Check the orientation/shape/version
        if self.affine.shape != (4, 4):
            raise InvalidExtensionError('Affine has incorrect shape')
        slice_dim = self.slice_dim
        if slice_dim != None:
            if not (0 <= slice_dim < 3):
                raise InvalidExtensionError('Slice dimension is not valid')
        if not (3 <= len(self.shape)):
            raise InvalidExtensionError('Shape is not valid')
            
        #Check all required meta dictionaries, make sure elements have correct
        #number of values
        valid_classes = self.get_valid_classes()
        for classification in valid_classes:
            if not classification in self._content:
                raise InvalidExtensionError('Missing required classification ' 
                                            '%s' % classification)
            cls_dict = self[classification]
            cls_n_vals = self.get_n_vals(classification)
            elif cls_n_vals > 1:
                for key, vals in cls_dict.iteritems():
                    n_vals = len(vals)
                    if n_vals != cls_n_vals:
                        msg = (('Incorrect number of values for key %s with '
                                'classification %s, expected %d found %d') %
                               (key, classes, cls_n_vals, n_vals)
                              )
                        raise InvalidExtensionError(msg)
                        
        #Check that all keys are uniquely classified
        for classification in valid_classes:
            for other_class in valid_classes:
                if classification == other_class:
                    continue
                intersect = (set(self[classification]) & 
                             set(self[other_class])
                            )
                if len(intersect) != 0:
                    raise InvalidExtensionError("One or more keys have "
                                                "multiple classifications")
            
    def get_all_keys(self):
        '''Get a list of all the meta data keys that are available across 
        classifications.'''
        keys = []
        for classification in self.get_valid_classes():
            keys += self[classification].keys()
        return keys

    def get_classification(self, key):
        '''Get the classification for the given `key`.
        
        Parameters
        ----------
        key : str
            The meta data key.
        
        Returns
        -------
        classification : tuple or None
            The classification tuple for the provided key or None if the key is 
            not found.
            
        '''
        for classification in self.get_valid_classes():
            if key in self[classification]:
                return classification
        return None
        
    def get_values(self, key):
        '''Get all values for the provided key. 

        Parameters
        ----------
        key : str
            The meta data key.
            
        Returns
        -------
        values 
             The value or values for the given key. The number of values 
             returned depends on the classification (see 'get_multiplicity').
        '''
        classification = self.get_classification(key)
        if classification is None:
            return None
        return self[classification][key]
        
    def filter_meta(self, filter_func):
        '''Filter the meta data.
        
        Parameters
        ----------
        filter_func : callable
            Must take a key and values as parameters and return True if they 
            should be filtered out.
            
        '''
        for classification in self.get_valid_classes():
            filtered = []
            curr_dict = self[classification]
            for key, values in curr_dict.iteritems():
                if filter_func(key, values):
                    filtered.append(key)
            for key in filtered:
                del curr_dict[key]
        
    def get_subset(self, dim, idx):
        '''Get a DcmMetaExtension containing a subset of the meta data 
        corresponding to a single index along a given dimension.
        
        Parameters
        ----------
        dim : int
            The dimension we are taking the subset along.
            
        idx : int
            The position on the dimension `dim` for the subset.
        
        Returns
        -------
        result : DcmMetaExtension
            A new DcmMetaExtension corresponding to the subset.
            
        '''
        shape = self.shape
        n_dim = len(shape)
        if not 0 <= dim < shape:
            raise ValueError("The argument 'dim' is out of range")
        valid_classes = self.get_valid_classes()
        
        #Make an empty extension for the result
        result_shape = list(shape)
        result_shape[dim] = 1
        while result_shape[-1] == 1 and len(result_shape) > 3:
            result_shape = result_shape[:-1]
        result = self.make_empty(result_shape, 
                                 self.affine,
                                 self.reorient_transform,
                                 self.slice_dim
                                )
        
        for src_class in valid_classes:
            #Constants always remain constant
            if src_class == 'const':
                for key, val in self['const'].iteritems():
                    result['const'][key] = deepcopy(val)
                continue
            
            if dim == self.slice_dim:
                if src_class != 'per_slice':
                    #Any per sample meta data stays the same
                    for key, vals in self[src_class].iteritems():
                        result[src_class][key] = deepcopy(vals)
                else:
                    #Per slice meta data gets reclassified
                    result._copy_slice(self, idx)
            elif dim < 3:
                #If splitting spatial, non-slice, dim keep everything
                for key, vals in self[src_class].iteritems():
                    result[src_class][key] = deepcopy(vals)
            elif dim >= 3:
                #Per slice/sample meta data may need to be reclassified
                result._copy_sample(self, src_class, dim, idx)
                
        return result
    
    #TODO: Update public methods and associated private helpers from here down
    def to_json(self):
        '''Return the extension encoded as a JSON string.'''
        self.check_valid()
        return self._mangle(self._content)
        
    @classmethod
    def from_json(klass, json_str):
        '''Create an extension from the JSON string representation.'''
        result = klass(dcm_meta_ecode, json_str)
        result.check_valid()
        return result
    
    @classmethod
    def make_empty(klass, shape, affine, reorient_transform=None, 
                   slice_dim=None):
        '''Make an empty DcmMetaExtension.
        
        Parameters
        ----------
        shape : tuple
            The shape of the data associated with this extension.
            
        affine : array
            The RAS affine for the data associated with this extension.
            
        reorient_transform : array
            The transformation matrix representing any reorientation of the 
            data array.
            
        slice_dim : int
            The index of the slice dimension for the data associated with this 
            extension
            
        Returns
        -------
        result : DcmMetaExtension
            An empty DcmMetaExtension with the required values set to the 
            given arguments.
        
        '''
        result = klass()
        result['const'] = OrderedDict()
        result['per_slice'] = OrderedDict()
        
        if len(shape) > 3 and shape[3] != 1:
            result._content['time'] = OrderedDict()
            result._content['time']['samples'] = OrderedDict()
            result._content['time']['slices'] = OrderedDict()
            
        if len(shape) > 4:
            result._content['vector'] = OrderedDict()
            result._content['vector']['samples'] = OrderedDict()
            result._content['vector']['slices'] = OrderedDict()
        
        result._content['dcmmeta_shape'] = []
        result.shape = shape
        result.affine = affine
        result.reorient_transform = reorient_transform
        result.slice_dim = slice_dim
        result.version = _meta_version
        
        return result
        
    @classmethod
    def from_sequence(klass, seq, dim, affine=None, slice_dim=None):
        '''Create an extension from a sequence of extensions.
        
        Parameters
        ----------
        seq : sequence
            The sequence of DcmMetaExtension objects.
        
        dim : int
            The dimension to merge the extensions along.
        
        affine : array
            The affine to use in the resulting extension. If None, the affine 
            from the first extension in `seq` will be used.
            
        slice_dim : int
            The slice dimension to use in the resulting extension. If None, the 
            slice dimension from the first extension in `seq` will be used.
        
        Returns
        -------
        result : DcmMetaExtension
            The result of merging the extensions in `seq` along the dimension
            `dim`.
        '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")
        
        n_inputs = len(seq)
        first_input = seq[0]
        input_shape = first_input.shape
        
        if len(input_shape) > dim and input_shape[dim] != 1:
            raise ValueError("The dim must be singular or not exist for the "
                             "inputs.")
        
        output_shape = list(input_shape)
        while len(output_shape) <= dim:
            output_shape.append(1)
        output_shape[dim] = n_inputs
        
        if affine is None:
            affine = first_input.affine
        if slice_dim is None:
            slice_dim = first_input.slice_dim
            
        result = klass.make_empty(output_shape, 
                                  affine, 
                                  None, 
                                  slice_dim)
        
        #Need to initialize the result with the first extension in 'seq'
        result_slc_norm = result.slice_normal
        first_slc_norm = first_input.slice_normal
        use_slices = (not result_slc_norm is None and
                      not first_slc_norm is None and 
                      np.allclose(result_slc_norm, first_slc_norm))
        for classes in first_input.get_valid_classes():
            if classes[1] == 'slices' and not use_slices:
                continue
            result._content[classes[0]][classes[1]] = \
                deepcopy(first_input.get_class_dict(classes))
        
        #Adjust the shape to what the extension actually contains
        shape = list(result.shape)
        shape[dim] = 1
        result.shape = shape

        #Initialize reorient transform
        reorient_transform = first_input.reorient_transform
        
        #Add the other extensions, updating the shape as we go
        for input_ext in seq[1:]:
            #If the affines or reorient_transforms don't match, we set the 
            #reorient_transform to None as we can not reliably use it to update 
            #directional meta data
            if ((reorient_transform is None or 
                 input_ext.reorient_transform is None) or 
                not (np.allclose(input_ext.affine, affine) or 
                     np.allclose(input_ext.reorient_transform, 
                                 reorient_transform)
                    )
               ):
                reorient_transform = None
            result._insert(dim, input_ext)
            shape[dim] += 1
            result.shape = shape
            
        #Set the reorient transform 
        result.reorient_transform = reorient_transform
            
        #Try simplifying any keys in global slices
        for key in result.get_class_dict(('global', 'slices')).keys():
            result._simplify(key)
            
        return result
        
    def __eq__(self, other):
        if not np.allclose(self.affine, other.affine):
            return False
        if self.shape != other.shape:
            return False
        if self.slice_dim != other.slice_dim:
            return False
        if self.version != other.version:
            return False
        for classes in self.get_valid_classes():
            if (dict(self.get_class_dict(classes)) != 
               dict(other.get_class_dict(classes))):
                return False
                
        return True
    
    def _simplify(self, key):
        '''Try to simplify (reduce the number of values) of a single meta data 
        element by changing its classification. Return True if the 
        classification is changed, otherwise False. 
        
        Looks for values that are constant or repeating with some pattern. 
        Constant elements with a value of None will be deleted.
        '''
        curr_class = self.get_classification(key)
        values = self[curr_class][key]
        
        if curr_class == 'const':
            #If the class is const then just delete it if the value is None
            if not values is None:
                return False
        elif is_constant(values):
            #Check if globaly constant
            self['const'][key] = values[0]
        else:
            #Check if we can move to a (higher) per sample classification
            shape = self.shape
            n_dims = len(shape)
            if n_dims < 4:
                return False
            
            if curr_class == 'per_slice':
                first_dim = 3
            else:
                first_dim = self.get_per_sample_dim(curr_class) + 1
                if first_dim == n_dims:
                    return False
                    
            for dest_dim in xrange(n_dims-1, first_dim-1, -1): 
                dest_class = 'per_sample_%d' % dest_dim
                period = self.get_n_vals(curr_class) / self.get_n_vals(dest_class)
                if is_constant(values, period):
                    self[dest_cls][key] = values[::period]
                    break
            else:
                return False
            
        del self.get_class_dict(curr_class)[key]
        return True
        
    _preserving_changes = {None : (('global', 'const'),
                                   ('vector', 'samples'),
                                   ('time', 'samples'),
                                   ('time', 'slices'),
                                   ('vector', 'slices'),
                                   ('global', 'slices'),
                                  ),
                           ('global', 'const') : (('vector', 'samples'),
                                                  ('time', 'samples'),
                                                  ('time', 'slices'),
                                                  ('vector', 'slices'),
                                                  ('global', 'slices'),
                                                 ),
                           ('vector', 'samples') : (('time', 'samples'),
                                                    ('global', 'slices'),
                                                   ),
                           ('time', 'samples') : (('global', 'slices'),
                                                 ),
                           ('time', 'slices') : (('vector', 'slices'),
                                                 ('global', 'slices'),
                                                ),
                           ('vector', 'slices') : (('global', 'slices'),
                                                  ),
                           ('global', 'slices') : tuple(),
                          }
    '''Classification mapping showing allowed changes when increasing the 
    multiplicity.'''        
        
    def _get_changed_class(self, key, new_class, n_slices=None):
        '''Get an array of values corresponding to a single meta data 
        element with its classification changed by increasing its 
        multiplicity. This will preserve all the meta data and allow easier
        merging of values with different classifications.'''
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return values
        #print key
        #print curr_class
        #print new_class    
        if not new_class in self._preserving_changes[curr_class]:
            raise ValueError("Classification change would lose data.")
        
        if curr_class is None:
            curr_mult = 1
            per_slice = False
        else:
            curr_mult = self.get_multiplicity(curr_class)
            per_slice = curr_class[1] == 'slices'
        if new_class[1] == 'slices' and not n_slices is None:
            new_mult = n_slices
        elif new_class in self.get_valid_classes():
            new_mult = self.get_multiplicity(new_class)
        else:
            new_mult = 1
        
        mult_fact = new_mult / curr_mult
        if curr_mult == 1:
            values = [values] 
            
            
        if per_slice:
            result = values * mult_fact
        else:
            result = []
            for value in values:
                result.extend([deepcopy(value)] * mult_fact)
            
        if new_class == ('global', 'const'):
            result = result[0]
            
        return result
        
        
    def _change_class(self, key, new_class):
        '''Change the classification of the meta data element in place. See 
        _get_changed_class.'''
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return

        self.get_class_dict(new_class)[key] = self._get_changed_class(key, 
                                                                      new_class)
        
        if not curr_class is None:
            del self.get_class_dict(curr_class)[key]
    
    
    
    def _copy_slice(self, other, idx):
        '''Get a copy of some per slice meta data corresponding to a single 
        slice index.'''
        shape = self.shape
        n_dim = len(shape)
        
        #If we have extra_spatial dimensions reclassify to be per_sample_n 
        #where n is the first non-singular dim. Otherwise reclassify as const.
        dest_class = None
        if n_dim > 3:
            for dim_idx in xrange(3, n_dim):
                if shape[dim_idx] != 1:
                    dest_class = 'per_sample_%d' % dim_idx
                    break
        if dest_class is None:
            dest_class = 'const'
           
        #Copy a subset of the meta data to the dest_class and try to simplify
        src_dict = other['per_slice']
        dest_dict = self[dest_class]
        dest_n_vals = self.get_n_vals(dest_class)
        stride = other.n_slices
        for key, vals in src_dict.iteritems():
            subset_vals = vals[idx::stride]
            if len(subset_vals) == 1:
                subset_vals = subset_vals[0]
            dest_dict[key] = deepcopy(subset_vals)
            self._simplify(key)

    def _per_slice_subset(self, key, subset_dim, idx):
        '''Get a subset of the meta data values with the classificaion 
        'per_slice' corresponding to a single sample along the extra spatial
        dimension 'sub_dim'.
        '''
        n_slices = self.n_slices
        shape = self.shape
        n_dim = len(shape)
        vals = src_dict = self['per_slice'][key]
        slices_per_sample = [n_slices]
        for dim xrange(4, n_dim):
            slices_per_sample.append(slices_per_sample[-1] * shape[dim])
        
        #If the subset dim is the last one we can grab a contiguous block
        if subset_dim == n_dim - 1:
            start_idx = idx * slices_per_sample[-1]
            end_idx = start_idx + slices_per_sample[-1]
            result = vals[start_idx:end_idx]
        else:
            #The meta data for this sample is spread out across higher 
            #dimesnions
            result = []
            iters = [xrange(dim_size) for dim_size in shape[subset_dim + 1:]]
            for hyper_idices in itertools.product(*iters):
                start_idx = 0
                for idx_offset, hyper_index in enumerate(hyper_indices):
                    start_idx += (hyper_index * 
                                  slices_per_sample[subset_dim + idx_offset + 1])
                end_idx = start_idx + slices_per_sample[subset_dim]
                result += vals[start_idx:end_idx]
        
        return result
    
    def _copy_sample(self, other, src_class, dim, idx):
        '''Get a copy of meta data from 'other' instance with classification 
        'src_class' corresponding to one index along the given extra spatial 
        dimension. The source class must be per slice or per sample.'''
        src_dict = other[src_class]
        if src_class == 'per_slice':
            #Take a subset of per_slice meta and try to simplify
            for key, vals in src_dict.iteritems():
                subset_vals = \
                    other._global_slice_subset(key, dim, idx)
                self[src_class][key] = deepcopy(subset_vals)
                self._simplify(key)
        else:
            #If we are indexing on the same dim as the src_class we need to 
            #change the classification
            src_dim = other.get_per_sample_dim(src_class)
            if src_dim == dim:
                #Reclassify to the next non-singular extra spatial dim, or const
                dest_class = None
                shape = self.shape
                for dest_dim in xrange(dim, len(shape)):
                    if shape[dest_dim] != 1:
                        dest_class = 'per_sample_%d' % dest_dim
                        break
                else:
                    dest_class  = 'const'
                
                if dest_class == 'const':
                    for key, vals in src_dict.iteritems():
                        self['const'][key] = deepcopy(vals[idx])
                else:
                    stride = other.shape[src_dim]
                    for key, vals in src_dict.iteritems():
                        self[dest_cls][key] = deepcopy(vals[idx::stride])
                    for key in src_dict.keys():
                        self._simplify(key)
                    
            else: 
                #Otherwise classification does not change
                #The multiplicity will change if the src_dim is less than dim 
                #we are taking the subset along
                if src_dim < dim:
                    dest_mult = self.get_multiplicity(src_class)
                    start_idx = idx * dest_mult
                    end_idx = start_idx + dest_mult
                    for key, vals in src_dict.iteritems():
                        self[src_class][key] = \
                            deepcopy(vals[start_idx:end_idx])
                        self._simplify(key)
                else: #Otherwise multiplicity is unchanged
                    for key, vals in src_dict.iteritems():
                        self[src_class][key] = deepcopy(vals)
       
    def _insert(self, dim, other):
        '''Insert the meta data from 'other' as if we were appending 
        the corresponding data set along 'dim'.'''
        missing_keys = list(set(self.get_keys()) - set(other.get_keys()))
        
        self_slc_norm = self.slice_normal
        other_slc_norm = other.slice_normal
        use_slices = (not self_slc_norm is None and
                      not other_slc_norm is None and 
                      np.allclose(self_slc_norm, other_slc_norm))
        if not use_slices:
            #If not using slices we add any keys to missing_keys
            for other_classes in other.get_valid_classes():
                if other_classes[1] == 'slices':
                    missing_keys += other.get_class_dict(other_classes).keys()
        
        for other_classes in other.get_valid_classes():
            if other_classes[1] == 'slices' and not use_slices: 
                continue
            
            other_keys = other.get_class_dict(other_classes).keys()
            
            #Treat missing keys as if they were in global const and have a value
            #of None
            if other_classes == ('global', 'const'):
                other_keys += missing_keys
            
            #When possible, reclassify our meta data so it matches the other
            #classificatoin
            for key in other_keys:
                local_classes = self.get_classification(key)
                if local_classes != other_classes:
                    local_allow = self._preserving_changes[local_classes]
                    other_allow = self._preserving_changes[other_classes]
                    if other_classes in local_allow:
                        self._change_class(key, other_classes)
                    elif not local_classes in other_allow:
                        best_dest = None
                        for dest_class in local_allow:
                            if (dest_class[0] in self._content and 
                               dest_class in other_allow):
                                best_dest = dest_class
                                break
                        self._change_class(key, best_dest)
                        
            #Insert new meta data and further reclassify as necessary
            for key in other_keys:
                if dim == self.slice_dim:
                    self._insert_slice(key, 
                                       other, 
                                       other_classes)
                elif dim < 3:
                    self._insert_non_slice(key, 
                                           other, 
                                           other_classes)
                elif dim == 3:
                    self._insert_sample(key, 
                                        other, 
                                        'time', 
                                        other_classes)
                elif dim == 4:
                    self._insert_sample(key, 
                                        other, 
                                        'vector', 
                                        other_classes)
               
    def _insert_slice(self, key, other, other_classes):
        '''Insert the element corresponding to 'key' from 'other' as 
        if we were appending the corresponding data set along the slice 
        dimension.'''
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, 
                                              classes, 
                                              other_classes,
                                              self.slice_dim)

        #Handle some common / simple insertions with special cases
        if classes == ('global', 'const'):
            if local_vals != other_vals:
                for dest_base in ('time', 'vector', 'global'):
                    if dest_base in self._content:
                        self._change_class(key, (dest_base, 'slices'))
                        other_vals = other._get_changed_class(key, 
                                                              (dest_base, 
                                                               'slices'),
                                                              other_classes,
                                                              self.slice_dim)
                        self.get_values(key).extend(other_vals)
                        break
        elif classes == ('time', 'slices'):
            local_vals.extend(other_vals)
        else:
            #Default to putting in global slices and simplifying later
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_class_dict(('global', 'slices'))[key]
                other_vals = other._get_changed_class(key, 
                                                      ('global', 'slices'),
                                                      other_classes,
                                                      self.slice_dim)
            
            #Need to interleave slices from different volumes
            n_slices = self.n_slices
            shape = self.shape
            n_vols = 1
            for dim_size in shape[3:]:
                n_vols *= dim_size
            
            intlv = []
            loc_start = 0
            oth_start = 0
            for vol_idx in xrange(n_vols):
                intlv += local_vals[loc_start:loc_start + n_slices]
                intlv += other_vals[oth_start:oth_start + 1]
                loc_start += n_slices
                oth_start += 1
                
            self.get_class_dict(('global', 'slices'))[key] = intlv
            
    def _insert_non_slice(self, key, other, other_classes):
        '''Insert element corresponding to 'key' as if we were 
        appending the corresponding data set along a spatial dimension 
        that is not the slice dim.
        '''
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, 
                                              classes, 
                                              other_classes,
                                              self.slice_dim)
        
        #Cannot capture variations along non-slice dim, just delete
        if local_vals != other_vals:
            del self.get_class_dict(classes)[key]
            
    def _insert_sample(self, key, other, sample_base, other_classes):
        '''Insert element corresponding to 'key' as if we were 
        appending the corresponding data set along the time/vector 
        dimension.
        '''
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, 
                                              classes, 
                                              other_classes, 
                                              self.slice_dim)
            
        if classes == ('global', 'const'):
            if local_vals != other_vals:
                self._change_class(key, (sample_base, 'samples'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key, 
                                                      (sample_base, 
                                                       'samples'),
                                                      other_classes,
                                                      self.slice_dim)
                local_vals.extend(other_vals)
        elif classes == (sample_base, 'samples'):
            local_vals.extend(other_vals)
        elif classes[1] == 'samples':
            #Need to handle this case seperately, as we might not be 
            #able to demote to global slices and then simplify as is 
            #done in the last case (slice dim might be undefined)
        else:
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key, 
                                                      ('global', 
                                                       'slices'),
                                                      other_classes,
                                                      self.slice_dim)
            
            shape = self.shape
            n_dims = len(shape)
            if sample_base == 'time' and n_dims == 5:
                #Need to interleave values from the time points in each 
                #vector component
                
                n_slices = self.n_slices #This could be None? If we are in the "else" clause then the meta key must be per slice...
                slices_per_vec = n_slices * shape[3]
                oth_slc_per_vec = n_slices * other.shape[3]
                
                intlv = []
                loc_start = 0
                oth_start = 0
                for vec_idx in xrange(shape[4]):
                    intlv += local_vals[loc_start:loc_start+slices_per_vec]
                    intlv += other_vals[oth_start:oth_start+oth_slc_per_vec]
                    loc_start += slices_per_vec
                    oth_start += oth_slc_per_vec
                    
                self.get_class_dict(('global', 'slices'))[key] = intlv
            else:
                local_vals.extend(other_vals)
            
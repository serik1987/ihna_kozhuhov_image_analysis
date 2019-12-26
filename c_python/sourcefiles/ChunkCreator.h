//
// Created by serik1987 on 24.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_CHUNKCREATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_CHUNKCREATOR_H

extern "C"{

    /**
     * Creates the chunk from the handle. The chunk will have an appropriate class
     *
     * @param handle the chunk handle
     * @param parent - the parent object
     * @return pointer to the chunk object or NULL if such a chunk type is not recognized
     */
    static PyImanS_ChunkObject* PyImanS_Chunk_FromHandle(void* handle, PyObject* parent){
        using namespace GLOBAL_NAMESPACE;

        auto* pchunk = (Chunk*)handle;
        PyImanS_ChunkObject* object;
        PyObject* type_object;
        if (pchunk->getName() == "ISOI"){
            type_object = (PyObject*)&PyImanS_IsoiChunkType;
        } else if (pchunk->getName() == "SOFT") {
            type_object = (PyObject *) &PyImanS_SoftChunkType;
        } else if (pchunk->getName() == "COMP") {
            type_object = (PyObject *) &PyImanS_CompChunkType;
        } else if (pchunk->getName() == "COST") {
            type_object = (PyObject *) &PyImanS_CostChunkType;
        } else if (pchunk->getName() == "DATA") {
            type_object = (PyObject *) &PyImanS_DataChunkType;
        } else if (pchunk->getName() == "EPST") {
            type_object = (PyObject *) &PyImanS_EpstChunkType;
        } else if (pchunk->getName() == "GREE") {
            type_object = (PyObject *) &PyImanS_GreenChunkType;
        } else if (pchunk->getName() == "HARD") {
            type_object = (PyObject *) &PyImanS_HardChunkType;
        } else if (pchunk->getName() == "ROIS") {
            type_object = (PyObject *) &PyImanS_RoisChunkType;
        } else if (pchunk->getName() == "SYNC") {
            type_object = (PyObject*) &PyImanS_SyncChunkType;
        } else {
            PyErr_SetString(PyExc_NotImplementedError, "The chunk is not still supported");
            return NULL;
        }
        object = (PyImanS_ChunkObject*)PyObject_CallFunction(type_object, "O", parent);
        if (object == NULL){
            return NULL;
        }
        object->handle = handle;

        return object;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_CHUNKCREATOR_H

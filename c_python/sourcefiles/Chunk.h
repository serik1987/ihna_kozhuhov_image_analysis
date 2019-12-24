//
// Created by serik1987 on 24.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_CHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_CHUNK_H

extern "C"{

    typedef struct {
        PyObject_HEAD
        void* handle;
        PyObject* parent;
    } PyImanS_ChunkObject;

    static PyTypeObject PyImanS_ChunkType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles.Chunk",
        .tp_basicsize = sizeof(PyImanS_ChunkObject),
        .tp_itemsize = 0,
    };

    static PyImanS_ChunkObject* PyImanS_Chunk_New(PyTypeObject* type, PyObject*, PyObject*){
        printf("SO Creating new Chunk object\n");
        auto* self = (PyImanS_ChunkObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->handle = NULL;
            self->parent = NULL;
        }
        return self;
    }

    static int PyImanS_Chunk_InitArgs(PyObject* args, const char* name, void** phandle, PyObject** pparent){
        printf("SO Chunk initialization\n");
        using namespace GLOBAL_NAMESPACE;
        PyObject* suggesting_parent;
        Chunk* suggesting_chunk;

        if (PyArg_ParseTuple(args, "")){
            uint32_t chunk_id = *(uint32_t*)name;
            int chunk_index;
            for (chunk_index = 0; chunk_index < ChunkHeader::CHUNK_CODE_NUMBER; ++chunk_index){
                if (ChunkHeader::CHUNK_CODE_LIST[chunk_index] == chunk_id){
                    break;
                }
            }
            if (chunk_index == ChunkHeader::CHUNK_CODE_NUMBER){
                PyErr_SetString(PyExc_ValueError, "This chunk is still unsupported by the Python API");
                return -1;
            }
            uint32_t chunk_size = ChunkHeader::CHUNK_SIZE_LIST[chunk_index];
            ChunkHeader header(name, chunk_size);
            suggesting_chunk = header.createChunk();
            *phandle = suggesting_chunk;
            *pparent = NULL;
            return 0;
        }

        PyErr_Clear();

        if (PyArg_ParseTuple(args, "O!", &PyImanS_SourceFileType, &suggesting_parent)){
            Py_INCREF(suggesting_parent);
            *pparent = suggesting_parent;
            *phandle = suggesting_parent;
            return 0;
        }

        return -1;
    }

    static int PyImanS_Chunk_Init(PyImanS_ChunkObject* self, PyObject* args, PyObject* kwds){
        PyErr_SetString(PyExc_NotImplementedError,
                "The Chunk class is purely abstract. Use any of its derived classes");

        return -1;
    }

    static void PyImanS_Chunk_Destroy(PyImanS_ChunkObject* self){
        using namespace GLOBAL_NAMESPACE;
        printf("SO Destruction of Chunk\n");

        if (self->parent != NULL){
            Py_DECREF(self->parent);
            self->handle = NULL;
        } else {
            auto* pchunk = (Chunk*)self->handle;
            delete pchunk;
        }

        Py_TYPE(self)->tp_free(self);
    }

    static int PyImanS_Chunk_Create(PyObject* module){

        PyImanS_ChunkType.tp_doc = "This is the base class for all file chunks.\n"
                                   "The whole content of the IMAN file is physically splitted into \n"
                                   "several pieces called 'chunks'. Each chunk represents an information \n"
                                   "that is logically grouped into some section. Any chunk is identified \n"
                                   "by so called 'chunk identifier' - a four-letter name of the chunk \n"
                                   "The chunk identifier explains what kind of data is stored within the chunk \n"
                                   "For instance: \n"
                                   "ISOI - stores all other chunks containing in the file\n"
                                   "SDFT - stores general information about frames and neighbour files in the \n"
                                   "file train\n"
                                   "HARD - stores properties of the experimental setup\n"
                                   "COST - stores general information about the stimulation protocol when \n"
                                   "continuous stimulation protocol is applied\n"
                                   "EPST - stores general information about the stimulation protocol when\n"
                                   "episodic stimulation about stimulation protocol is applied\n"
                                   "COMP - this chunk is presented in compressed files only and stores information \n"
                                   "that is important for the data decompression\n"
                                   "DATA - contains the intrinsic imaging signal itself \n"
                                   "GREE - available in Green Maps only\n"
                                   "ROIS, SYNC - no idea\n"
                                   "\n"
                                   "The chunk object allows to manager the data stored in a single chunk\n"
                                   "The ISOI chunk object manages all other chunks\n"
                                   "All chunks have properties that represent certain data stored within the chunk\n"
                                   "To access any chunk property please, use an index notation like:\n"
                                   "chunk['property_name']\n"
                                   "\n"
                                   "The following chunk properties are present in all chunks\n"
                                   "'ID' - the chunk ID (write chunk['ID'] to access)\n"
                                   "'size' - the space occupied by the chunk body on the hard disk. For ISOI or DATA \n"
                                   "chunks this value may be 0 that doesn't correspond to the actual chunk size\n"
                                   "Any other properties are specific for a certain chunk. You may find them by writing: \n"
                                   "help(chunk)\n"
                                   "To print all chunk properties, please, write:\n"
                                   "print(chunk)\n"
                                   "The following command will receive all chunk properties as a single string:\n"
                                   "str(chunk)";
        PyImanS_ChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_ChunkType.tp_new = (newfunc)PyImanS_Chunk_New;
        PyImanS_ChunkType.tp_init = (initproc)PyImanS_Chunk_Init;
        PyImanS_ChunkType.tp_dealloc = (destructor)PyImanS_Chunk_Destroy;

        if (PyType_Ready(&PyImanS_ChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_ChunkType);

        if (PyModule_AddObject(module, "_sourcefiles_Chunk", (PyObject*)&PyImanS_ChunkType) < 0){
            Py_DECREF(&PyImanS_ChunkType);
            return -1;
        }

        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_ChunkType;

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_CHUNK_H

//
// Created by serik1987 on 22.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_SOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_SOURCEFILE_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        void* file_handle;
    } PyImanS_SourceFileObject;

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

    typedef struct{
        PyImanS_ChunkObject super;
    } PyImanS_SoftChunkObject;

    static PyTypeObject PyImanS_SoftChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._SoftChunk",
            .tp_basicsize = sizeof(PyImanS_SoftChunkObject),
            .tp_itemsize = 0,
    };

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_IsoiChunkObject;

    static PyTypeObject PyImanS_IsoiChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._IsoiChunk",
            .tp_basicsize = sizeof(PyImanS_IsoiChunkObject),
            .tp_itemsize = 0,
    };

    static PyImanS_SourceFileObject* PyImanS_SourceFile_New(PyTypeObject* cls, PyObject* args, PyObject* kwds){
        PyImanS_SourceFileObject* self = NULL;
        self = (PyImanS_SourceFileObject*)cls->tp_alloc(cls, 0);
        if (self != NULL){
            self->file_handle = NULL;
        }
        return self;
    }

    static int PyImanS_SourceFile_Init(PyImanS_SourceFileObject* self, PyObject* args, PyObject* kwds){
        const char* path = NULL;
        const char* file = NULL;

        if (!PyArg_ParseTuple(args, "ss", &path, &file)){
            return -1;
        }

        self->file_handle = new GLOBAL_NAMESPACE::SourceFile(path, file);
        return 0;
    }

    static void PyImanS_SourceFile_Destroy(PyImanS_SourceFileObject* self){
        if (self->file_handle != NULL) {
            auto *file = (GLOBAL_NAMESPACE::SourceFile *) self->file_handle;
            delete file;
            self->file_handle = NULL;
        }
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static PyObject* PyImanS_SourceFile_GetFilePath(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyUnicode_FromString(pfile->getFilePath().c_str());
    }

    static PyObject* PyImanS_SourceFile_GetFilename(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyUnicode_FromString(pfile->getFileName().c_str());
    }

    static PyObject* PyImanS_SourceFile_GetFullName(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyUnicode_FromString(pfile->getFullname().c_str());
    }

    static PyObject* PyImanS_SourceFile_IsOpened(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyBool_FromLong(pfile->isOpened());
    }

    static PyObject* PyImanS_SourceFile_IsLoaded(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyBool_FromLong(pfile->isLoaded());
    }

    static PyObject* PyImanS_SourceFile_GetFrameHeaderSize(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyLong_FromUnsignedLong(pfile->getFrameHeaderSize());
    }

    static PyObject* PyImanS_SourceFile_GetFileHeaderSize(PyImanS_SourceFileObject* self, void*){
        auto* pfile = (GLOBAL_NAMESPACE::SourceFile*)self->file_handle;
        return PyLong_FromUnsignedLong(pfile->getFileHeaderSize());
    }

    static PyObject* PyImanS_SourceFile_GetFileType(PyImanS_SourceFileObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto pfile = (SourceFile*)self->file_handle;
        SourceFile::FileType type = pfile->getFileType();
        if (type == SourceFile::AnalysisFile){
            return PyUnicode_FromString("analysis");
        } else if (type == SourceFile::GreenFile){
            return PyUnicode_FromString("green");
        } else if (type == SourceFile::CompressedFile){
            return PyUnicode_FromString("compressed");
        } else if (type == SourceFile::StreamFile){
            return PyUnicode_FromString("stream");
        } else {
            return PyUnicode_FromString("unknown");
        }
    }

    static PyObject* PyImanS_SourceFile_Open(PyImanS_SourceFileObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* pfile = (SourceFile*)self->file_handle;
        PyObject* result = NULL;

        try{
            pfile->open();
            result = Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            result = NULL;
        }

        return result;
    }

    static PyObject* PyImanS_SourceFile_LoadFileInfo(PyImanS_SourceFileObject* self, PyObject*, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* pfile = (SourceFile*)self->file_handle;
        PyObject* result = NULL;

        try{
            pfile->loadFileInfo();
            result = Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            result = NULL;
        }

        return result;
    }

    static PyObject* PyImanS_SourceFile_Close(PyImanS_SourceFileObject* self, PyObject*, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* pfile = (SourceFile*)self->file_handle;
        PyObject* result = NULL;

        try{
            pfile->close();
            result = Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            result = NULL;
        }

        return result;
    }

    static PyObject* PyImanS_SourceFile_Str(PyImanS_SourceFileObject* self){
        using namespace GLOBAL_NAMESPACE;
        auto* pfile = (SourceFile*)self->file_handle;
        PyObject* result = NULL;

        try{
            std::stringstream ss;
            ss << *pfile << std::endl;
            result = PyUnicode_FromString(ss.str().c_str());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
        }

        return result;
    }

    static PyImanS_SoftChunkObject* PyImanS_SourceFile_GetSoft(PyImanS_SourceFileObject* self){
        using namespace GLOBAL_NAMESPACE;
        auto* pfile = (SourceFile*)self->file_handle;
        SoftChunk* chunk = NULL;
        PyImanS_SoftChunkObject* chunkObject = NULL;

        try{
            chunk = &pfile->getSoftChunk();
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

        chunkObject = (PyImanS_SoftChunkObject*)PyObject_CallFunction(
                (PyObject*)&PyImanS_SoftChunkType, "O", (PyObject*)self);
        chunkObject->super.handle = chunk;

        return chunkObject;
    }

    static PyImanS_IsoiChunkObject* PyImanS_SourceFile_GetIsoiChunk(PyImanS_SourceFileObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* pfile = (SourceFile*)self->file_handle;

        try{
            IsoiChunk* chunk = &pfile->getIsoiChunk();
            auto* chunkObject = (PyImanS_IsoiChunkObject*)PyObject_CallFunction(
                    (PyObject*)&PyImanS_IsoiChunkType, "O", self);
            chunkObject->super.handle = chunk;
            return chunkObject;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyGetSetDef PyImanS_SourceFileProperties[] = {
            {(char*)"file_path", (getter)PyImanS_SourceFile_GetFilePath, NULL,
             (char*)"Full path to the file", NULL},
            {(char*)"filename", (getter)PyImanS_SourceFile_GetFilename, NULL,
             (char*)"File name", NULL},
            {(char*)"full_name", (getter)PyImanS_SourceFile_GetFullName, NULL,
             (char*)"Full name of the file that consists of the file path and the file name", NULL},
            {(char*)"is_opened", (getter)PyImanS_SourceFile_IsOpened, NULL,
             (char*)"True is the file is in the opened state", NULL},
            {(char*)"is_loaded", (getter)PyImanS_SourceFile_IsLoaded, NULL,
             (char*)"True is the file header has already been loaded", NULL},
            {(char*)"frame_header_size", (getter)PyImanS_SourceFile_GetFrameHeaderSize, NULL,
             (char*)"Size of the frame header in bytes or 4GB if the file is not loaded", NULL},
            {(char*)"file_header_size", (getter)PyImanS_SourceFile_GetFileHeaderSize, NULL,
             (char*)"Size of file header, in bytes", NULL},
            {(char*)"file_type", (getter)PyImanS_SourceFile_GetFileType, NULL,
             (char*)"One of the following values:\n"
                    "'analysis' - analysis file that stores final analysis results\n"
                    "'green' - green file\n"
                    "'stream' - native data in uncompressed mode\n"
                    "'compressed' - native data in the compressed mode\n"
                    "'unknown' - the file header is not loaded or such a type is not supported in this version of \n"
                    "the module"},
            {(char*)"soft", (getter)PyImanS_SourceFile_GetSoft, NULL,
                (char*)"returns the SOFT chunk of the file (with retaining)", NULL},
            {(char*)"isoi", (getter)PyImanS_SourceFile_GetIsoiChunk, NULL,
                (char*)"returns the ISOI chunk of the file (with retaining)", NULL},
            {NULL}
    };

    static PyMethodDef PyImanS_SourceFileMethods[] = {
            {"open", (PyCFunction)PyImanS_SourceFile_Open, METH_NOARGS, "Opens the file for reading"},
            {"load_file_info", (PyCFunction)PyImanS_SourceFile_LoadFileInfo, METH_NOARGS,
             "Load the file header"},
            {"close", (PyCFunction)PyImanS_SourceFile_Close, METH_NOARGS, "Closes the file"},
            {NULL}
    };

    static PyTypeObject PyImanS_SourceFileType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._SourceFile",
        .tp_basicsize = sizeof(PyImanS_SourceFileObject),
        .tp_itemsize = 0,
    };

    static int PyImanS_SourceFile_Create(PyObject* module){

        PyImanS_SourceFileType.tp_doc = "For internal use. Apply SourceFile or any other derived class instead";
        PyImanS_SourceFileType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_SourceFileType.tp_new = (newfunc)PyImanS_SourceFile_New;
        PyImanS_SourceFileType.tp_dealloc = (destructor)PyImanS_SourceFile_Destroy;
        PyImanS_SourceFileType.tp_init = (initproc)PyImanS_SourceFile_Init;
        PyImanS_SourceFileType.tp_getset = PyImanS_SourceFileProperties;
        PyImanS_SourceFileType.tp_methods = PyImanS_SourceFileMethods;
        PyImanS_SourceFileType.tp_str = (reprfunc)PyImanS_SourceFile_Str;

        if (PyType_Ready(&PyImanS_SourceFileType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_SourceFileType);

        if (PyModule_AddObject(module, "_sourcefiles_SourceFile", (PyObject*)&PyImanS_SourceFileType) < 0){
            Py_DECREF(&PyImanS_SourceFileType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOURCEFILE_H

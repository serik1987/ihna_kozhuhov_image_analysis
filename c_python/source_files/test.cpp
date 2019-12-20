//
// Created by serik1987 on 19.12.2019.
//

#include "Python.h"
#include "exceptions.h"
#include "../../cpp/source_files/CompressedFileTrain.h"

/*
 * Needs for the purpose of this test module. In any other modules such a string shall be omitted
 */
namespace ihna::kozhukhov::image_analysis{
    const int SourceFile::CHUNK_ID_SIZE = ChunkHeader::CHUNK_ID_SIZE;
}

extern "C" {

    static PyObject* c_api_test(PyObject* module, PyObject* args){
        PyObject* list = PyList_New(0);
        if (PyList_Append(list, Py_BuildValue("")) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ImanError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_IoError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_TrainError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ExperimentModeError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_SynchronizationChannelNumberError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_UnsupportedExperimentModeError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_SourceFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FrameNumberError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_DataChunkNotFoundError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_IsoiChunkSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ExperimentChunkMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileHeaderMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FrameHeaderMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FrameDimensionsMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_DataTypeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_CompChunkNotExistError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileNotOpenedError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileOpenError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileReadError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ChunkError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_UnsupportedChunkError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ChunkSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ChunkNotFoundError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_IsoiChunkNotFoundError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileNotLoadedError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_DataChunkSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotAnalysisFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotGreenFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotCompressedFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotStreamFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotTrainHeadError) < 0) return NULL;
        return list;
    }

    static PyObject* connection_test(PyObject* module, PyObject* args){
        int exception_number;
        if (!PyArg_ParseTuple(args, "i", &exception_number)){
            return NULL;
        }
        if (exception_number == 0) {
            printf("iman_exception: \n");
            GLOBAL_NAMESPACE::iman_exception e("Sample message");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 1){
            printf("io_exception: \n");
            GLOBAL_NAMESPACE::io_exception e("Sample message", "FILE001.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 2){
            printf("train_exception: \n");
            GLOBAL_NAMESPACE::FileTrain::train_exception e("Sample message", "TRAIN002.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 3){
            printf("experiment_mode_exception: \n");
            GLOBAL_NAMESPACE::FileTrain::experiment_mode_exception e("TRAIN003.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 4){
            printf("synchronization_channel_number_exception: \n");
            GLOBAL_NAMESPACE::FileTrain::synchronization_channel_number_exception e("TRAIN004.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 5){
            printf("unsupported_experiment_mode_exception: \n");
            GLOBAL_NAMESPACE::FileTrain::unsupported_experiment_mode_exception e("TRAIN005.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 6){
            printf("source_file_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::source_file_exception e("Some message", "FILE006.DAT", "TRAIN006.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 7){
            printf("frame_number_exception: \n");
            GLOBAL_NAMESPACE::FileTrain::frame_number_mismatch e("FILE007.DAT", "TRAIN007.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 8){
            printf("data_chunk_size_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::data_chunk_size_mismatch e("FILE008.DAT", "TRAIN008.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 9){
            printf("isoi_chunk_size_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::isoi_chunk_size_mismatch e("FILE009.DAT", "TRAIN009.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 10){
            printf("file_size_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::file_size_mismatch e("FILE010.DAT", "TRAIN010.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 11){
            printf("experimental_chunk_not_found: \n");
            GLOBAL_NAMESPACE::FileTrain::experimental_chunk_not_found e("FILE011.DAT", "TRAIN011.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 12){
            printf("file_header_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::file_header_mismatch e("FILE012.DAT", "TRAIN012.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 13){
            printf("frame_header_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::frame_header_mismatch e("FILE013.DAT", "TRAIN013.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 14){
            printf("map_dimensions_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::map_dimensions_mismatch e("FILE014.DAT", "TRAIN014.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 15){
            printf("data_type_mismatch: \n");
            GLOBAL_NAMESPACE::FileTrain::data_type_mismatch e("FILE015.DAT", "TRAIN015.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 16){
            printf("comp_chunk_not_exist_exception: \n");
            GLOBAL_NAMESPACE::CompressedFileTrain::comp_chunk_not_exist_exception e("FILE016.DAT", "TRAIN016.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 17){
            printf("file_not_opened: \n");
            GLOBAL_NAMESPACE::SourceFile::file_not_opened e("F", "FILE017.DAT", "TRAIN017.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 18){
            printf("file_read_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::file_read_exception e("FILE018.DAT", "TRAIN018.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 19){
            printf("unsupported_chunk_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::unsupported_chunk_exception e("BLUE", "FILE019.DAT", "TRAIN019.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 20){
            printf("chunk_size_mismatch_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::chunk_size_mismatch_exception e("SOFT", "FILE020.DAT", "TRAIN020.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
         } else if (exception_number == 21){
            printf("chunk_not_found_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::chunk_not_found_exception e("COMP", "FILE021.DAT", "TRAIN021.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 22){
            printf("file_open_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::file_open_exception e("FILE022.DAT", "TRAIN022.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 23){
            printf("file_not_isoi_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::file_not_isoi_exception e("FILE023.DAT", "TRAIN023.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 24){
            printf("file_not_loaded_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::file_not_loaded_exception e("G", "FILE024.DAT", "TRAIN024.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 25){
            printf("data_chunk_not_found_exception: \n");
            GLOBAL_NAMESPACE::SourceFile::data_chunk_not_found_exception e("FILE025.DAT", "TRAIN025.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 26){
            printf("not_analysis_file_exception: \n");
            GLOBAL_NAMESPACE::AnalysisSourceFile::not_analysis_file_exception e("FILE026.DAT", "TRAIN026.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 27){
            printf("not_green_file_exception: \n");
            GLOBAL_NAMESPACE::GreenSourceFile::not_green_file_exception e("FILE027.DAT", "TRAIN027.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 28){
            printf("not_train_head: \n");
            GLOBAL_NAMESPACE::TrainSourceFile::not_train_head e("FILE028.DAT", "TRAIN028.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 29) {
            printf("not_stream_file: \n");
            GLOBAL_NAMESPACE::StreamSourceFile::not_stream_file e("FILE029.DAT", "TRAIN029.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 30) {
            printf("not_compressed_file_exception: \n");
            GLOBAL_NAMESPACE::CompressedSourceFile::not_compressed_file_exception e("FILE030.DAT", "TRAIN030.DAT");
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else if (exception_number == 31) {
            printf("data_chunk_read_exception: \n");
            GLOBAL_NAMESPACE::DataChunk::data_chunk_read_exception e;
            if (PyImanS_Exception_process(&e) < 0) {
                return NULL;
            }
        } else {
            printf("PyImanS_Exception_process returned OK\n");
            return Py_BuildValue("");
        }
    }

    static struct PyMethodDef test_methods[] = {
            {"c_api_test", c_api_test, METH_NOARGS, "C API test"},
            {"connection_test", connection_test, METH_VARARGS, "Connection test"},
            {NULL}
    };

    static struct PyModuleDef test_module = {
            PyModuleDef_HEAD_INIT,
            .m_name = "test",
            .m_doc = "this is a test module",
            .m_size = -1,
            .m_methods = test_methods
    };

    PyMODINIT_FUNC PyInit_test(void){
        PyObject* module = NULL;

        module = PyModule_Create(&test_module);
        if (module == NULL) return NULL;

        if (PyImanS_Import__exceptions() < 0){
            Py_DECREF(module);
            return NULL;
        }

        return module;
    }

}
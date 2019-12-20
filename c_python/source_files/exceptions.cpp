//
// Created by serik1987 on 19.12.2019.
//

#include "Python.h"
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOURCE_FILES_EXCEPTIONS_MODULE
#include "exceptions.h"
#include "../../cpp/source_files/CompressedFileTrain.h"

#define ERR_NAME(x) x, FULL_ERROR_NAME_PREFIX x

extern "C" {

    static const int CHUNK_ID_SIZE = GLOBAL_NAMESPACE::ChunkHeader::CHUNK_ID_SIZE;

    static PyObject *PyExc_ImanError = NULL;
    static PyObject* PyExc_ImanIoError = NULL;
    static PyObject* PyExc_TrainError = NULL;
    static PyObject* PyExc_ExperimentModeError = NULL;
    static PyObject* PyExc_SynchronizationChannelNumberError = NULL;
    static PyObject* PyExc_UnsupportedExperimentModeError = NULL;
    static PyObject* PyExc_SourceFileError = NULL;
    static PyObject* PyExc_FrameNumberError = NULL;
    static PyObject* PyExc_DataChunkSizeMismatchError = NULL;
    static PyObject* PyExc_IsoiChunkSizeMismatchError = NULL;
    static PyObject* PyExc_FileSizeMismatchError = NULL;
    static PyObject* PyExc_ExperimentChunkMismatchError = NULL;
    static PyObject* PyExc_FileHeaderMismatchError = NULL;
    static PyObject* PyExc_FrameHeaderMismatchError = NULL;
    static PyObject* PyExc_FrameDimensionsMismatchError = NULL;
    static PyObject* PyExc_DataTypeMismatchError = NULL;
    static PyObject* PyExc_CompChunkNotExistError = NULL;
    static PyObject* PyExc_FileOpenError = NULL;
    static PyObject* PyExc_FileReadError = NULL;
    static PyObject* PyExc_ChunkError = NULL;
    static PyObject* PyExc_UnsupportedChunkError = NULL;
    static PyObject* PyExc_ChunkSizeMismatchError = NULL;
    static PyObject* PyExc_ChunkNotFoundError = NULL;
    static PyObject* PyExc_FileNotOpenedError = NULL;
    static PyObject* PyExc_IsoiChunkNotFoundError = NULL;
    static PyObject* PyExc_FileNotLoadedError = NULL;
    static PyObject* PyExc_DataChunkNotFoundError = NULL;
    static PyObject* PyExc_NotAnalysisFileError = NULL;
    static PyObject* PyExc_NotGreenFileError = NULL;
    static PyObject* PyExc_NotTrainHeadError = NULL;
    static PyObject* PyExc_NotStreamFileError = NULL;
    static PyObject* PyExc_NotCompressedFileError = NULL;

    /**
     * Transforms the C++ exception into the Python exception
     *
     * @param exception pointer to the C++ exception. The exception shall be an instance of iman_exception.
     * @return 0 if no any exception is set, -1 otherwise
     */
    static int Exception_process(void* exception){
        if (exception == NULL) return 0;
        auto* base = (GLOBAL_NAMESPACE::iman_exception*)exception;
        auto* io = dynamic_cast<GLOBAL_NAMESPACE::io_exception*>(base);
        auto* train = dynamic_cast<GLOBAL_NAMESPACE::FileTrain::train_exception*>(base);
        auto* experiment_mode = dynamic_cast<GLOBAL_NAMESPACE::FileTrain::experiment_mode_exception*>(base);
        auto* synch_channel_number =
                dynamic_cast<GLOBAL_NAMESPACE::FileTrain::synchronization_channel_number_exception*>(base);
        auto* wrong_emode =
                        dynamic_cast<GLOBAL_NAMESPACE::FileTrain::unsupported_experiment_mode_exception*>(base);
        auto* source_file =
                dynamic_cast<GLOBAL_NAMESPACE::SourceFile::source_file_exception*>(base);
        auto* data_chunk_read =
                dynamic_cast<GLOBAL_NAMESPACE::DataChunk::data_chunk_read_exception*>(base);

        if (data_chunk_read != nullptr){
            PyErr_SetString(PyExc_NotImplementedError, base->what());
            return -1;
        }


        if (io == nullptr) {
            PyErr_SetString(PyExc_ImanError, base->what());
            return -1;
        }
        if (train != nullptr){
            PyObject_SetAttrString(PyExc_TrainError, "train_name", PyUnicode_FromString(train->getFilename().c_str()));
            if (experiment_mode != nullptr){
                PyErr_SetString(PyExc_ExperimentModeError, experiment_mode->what());
                return -1;
            }
            if (synch_channel_number != nullptr){
                PyErr_SetString(PyExc_SynchronizationChannelNumberError, base->what());
                return -1;
            }
            if (wrong_emode != nullptr){
                PyErr_SetString(PyExc_UnsupportedExperimentModeError, base->what());
                return -1;
            }
            PyErr_SetString(PyExc_TrainError, train->what());
            return -1;
        }
        if (source_file != nullptr){
            if (source_file->getTrainName() != ""){
                PyObject_SetAttrString(PyExc_TrainError, "train_name",
                        PyUnicode_FromString(source_file->getTrainName().c_str()));
            } else {
                PyObject_SetAttrString(PyExc_TrainError, "train_name", Py_BuildValue(""));
            }
            PyObject_SetAttrString(PyExc_SourceFileError, "file_name",
                                   PyUnicode_FromString(source_file->getFilename().c_str()));
            PyObject_SetAttrString(PyExc_ChunkError, "chunk_name", Py_BuildValue(""));
            auto* frame_number_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::frame_number_mismatch*>(base);
            auto* data_chunk_size_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::data_chunk_size_mismatch*>(base);
            auto* isoi_chunk_size_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::isoi_chunk_size_mismatch*>(base);
            auto* file_size_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::file_size_mismatch*>(base);
            auto* experimental_chunk_not_found_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::experimental_chunk_not_found*>(base);
            auto* file_header_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::file_header_mismatch*>(base);
            auto* frame_header_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::frame_header_mismatch*>(base);
            auto* map_dimensions_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::map_dimensions_mismatch*>(base);
            auto* data_type_mismatch_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::FileTrain::data_type_mismatch*>(base);
            auto* comp_chunk_not_exist =
                    dynamic_cast<GLOBAL_NAMESPACE::CompressedFileTrain::comp_chunk_not_exist_exception*>(base);
            auto* file_not_opened_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::file_not_opened*>(base);
            auto* file_read =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::file_read_exception*>(base);
            auto* unsupported_chunk =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::unsupported_chunk_exception*>(base);
            auto* chunk_size_mismatch =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::chunk_size_mismatch_exception*>(base);
            auto* chunk_not_found =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::chunk_not_found_exception*>(base);
            auto* file_open_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::file_open_exception*>(base);
            auto* file_not_isoi =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::file_not_isoi_exception*>(base);
            auto* file_not_loaded =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::file_not_loaded_exception*>(base);
            auto* data_chunk_not_found =
                    dynamic_cast<GLOBAL_NAMESPACE::SourceFile::data_chunk_not_found_exception*>(base);
            auto* not_analysis_file =
                    dynamic_cast<GLOBAL_NAMESPACE::AnalysisSourceFile::not_analysis_file_exception*>(base);
            auto* not_green_file =
                    dynamic_cast<GLOBAL_NAMESPACE::GreenSourceFile::not_green_file_exception*>(base);
            auto* not_train_head_exception =
                    dynamic_cast<GLOBAL_NAMESPACE::TrainSourceFile::not_train_head*>(base);
            auto* not_stream =
                    dynamic_cast<GLOBAL_NAMESPACE::StreamSourceFile::not_stream_file*>(base);
            auto* not_compressed =
                    dynamic_cast<GLOBAL_NAMESPACE::CompressedSourceFile::not_compressed_file_exception*>(base);

            if (frame_number_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_FrameNumberError, base->what());
                return -1;
            }
            if (data_chunk_size_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_DataChunkSizeMismatchError, base->what());
                return -1;
            }
            if (isoi_chunk_size_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_IsoiChunkSizeMismatchError, base->what());
                return -1;
            }
            if (file_size_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_FileSizeMismatchError, base->what());
                return -1;
            }
            if (experimental_chunk_not_found_exception != nullptr){
                PyErr_SetString(PyExc_ExperimentChunkMismatchError, base->what());
                return -1;
            }
            if (file_header_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_FileHeaderMismatchError, base->what());
                return -1;
            }
            if (frame_header_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_FrameHeaderMismatchError, base->what());
                return -1;
            }
            if (map_dimensions_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_FrameDimensionsMismatchError, base->what());
                return -1;
            }
            if (data_type_mismatch_exception != nullptr){
                PyErr_SetString(PyExc_DataTypeMismatchError, base->what());
                return -1;
            }
            if (comp_chunk_not_exist != nullptr){
                PyErr_SetString(PyExc_CompChunkNotExistError, base->what());
                return -1;
            }
            if (file_not_opened_exception != nullptr){
                PyErr_SetString(PyExc_FileNotOpenedError, base->what());
                return -1;
            }
            if (file_read != nullptr){
                PyErr_SetString(PyExc_FileReadError, base->what());
                return -1;
            }
            if (file_open_exception != nullptr){
                PyErr_SetString(PyExc_FileOpenError, base->what());
                return -1;
            }
            if (file_not_isoi != nullptr){
                PyErr_SetString(PyExc_IsoiChunkNotFoundError, base->what());
                return -1;
            }
            if (file_not_loaded != nullptr){
                PyErr_SetString(PyExc_FileNotLoadedError, base->what());
                return -1;
            }
            if (data_chunk_not_found != nullptr){
                PyErr_SetString(PyExc_DataChunkNotFoundError, base->what());
                return -1;
            }
            if (not_analysis_file != nullptr){
                PyErr_SetString(PyExc_NotAnalysisFileError, base->what());
                return -1;
            }
            if (not_green_file != nullptr){
                PyErr_SetString(PyExc_NotGreenFileError, base->what());
                return -1;
            }
            if (not_train_head_exception != nullptr){
                PyErr_SetString(PyExc_NotTrainHeadError, base->what());
                return -1;
            }
            if (not_stream != nullptr){
                PyErr_SetString(PyExc_NotStreamFileError, base->what());
                return -1;
            }
            if (not_compressed != nullptr){
                PyErr_SetString(PyExc_NotCompressedFileError, base->what());
                return -1;
            }
            const char* chunk_id = NULL;
            char chunk_id_supp[] = "\x00\x00\x00\x00";
            if (unsupported_chunk != nullptr){
                PyErr_SetString(PyExc_UnsupportedChunkError, base->what());
                chunk_id = unsupported_chunk->getChunkName();
                strncpy(chunk_id_supp, chunk_id, CHUNK_ID_SIZE);
            }
            if (chunk_size_mismatch != nullptr){
                PyErr_SetString(PyExc_ChunkSizeMismatchError, base->what());
                chunk_id = chunk_size_mismatch->getChunkName();
                strncpy(chunk_id_supp, chunk_id, CHUNK_ID_SIZE);
            }
            if (chunk_not_found != nullptr){
                PyErr_SetString(PyExc_ChunkNotFoundError, base->what());
                chunk_id = chunk_not_found->getChunkName();
                strncpy(chunk_id_supp, chunk_id, CHUNK_ID_SIZE);
            }
            if (chunk_id != NULL){
                PyObject_SetAttrString(PyExc_ChunkError, "chunk_name", PyUnicode_FromString(chunk_id_supp));
                return -1;
            }
            PyErr_SetString(PyExc_SourceFileError, base->what());
            return -1;
        }
        PyErr_SetString(PyExc_ImanIoError, io->what());
        return -1;
    }

    static struct PyMethodDef core_methods[] = {
            {NULL}
    };

    static struct PyModuleDef core = {
            PyModuleDef_HEAD_INIT,
            .m_name = "_exceptions",
            .m_doc = "This module is to import all exception that can be thrown during the generation of the code.",
            .m_size = -1,
            .m_methods = core_methods
    };

    /**
     * Imports a single exception to the module
     *
     * @param module pointer to the module (created by PyModule_Create function)
     * @param exception_name class name corresponds to the exception
     * @param exception_full_name full exception name like packagename.modulename.ClassName
     * @param exception_doc short exception documentation
     * @param base_exception reference to the base exception class
     * @param exception_object reference to the reference to the exception object. When the exception
     * will be successfully created its reference will be written to the *exception_object
     * @param var_name name of an ultimate exception parameter that is not inherited by the base exception
     * or NULL if there is no such parameters
     * @return 0 on success, -1 on failure
     */
    static int import_exception(PyObject* module,
            const char* exception_name,
            const char* exception_full_name, const char* exception_doc,
            PyObject* base_exception,
            PyObject** exception_object,
            const char* var_name){

        PyObject* variables = NULL;
        PyObject* value = NULL;

        if (var_name != NULL){
            variables = PyDict_New();
            if (variables == NULL){
                printf("Trying to create the dictionary\n");
                return -1;
            }

            value = Py_BuildValue("");
            if (value == NULL){
                printf("Try to create the dictionary value\n");
                Py_DECREF(variables);
                return -1;
            }

            if (PyDict_SetItemString(variables, var_name, value) < 0){
                printf("Try to set the dictionary item\n");
                Py_DECREF(variables);
                return -1;
            }
        }

        *exception_object = PyErr_NewExceptionWithDoc(exception_full_name, exception_doc, base_exception, variables);
        if (*exception_object == NULL){
            Py_XDECREF(variables);
            Py_XDECREF(value);
            printf("Error in creating the following exception: %s\n", exception_name);
            return -1;
        }
        if (PyModule_AddObject(module, exception_name, *exception_object) < 0){
            Py_XDECREF(variables);
            Py_XDECREF(value);
            Py_DECREF(*exception_object);
            printf("Error in adding the following exception to the object: %s\n", exception_name);
            return -1;
        }

        return 0;
    }

    /**
     * Creates all exceptions necessary for ihna.kozhukhov.imageanalysis.sourcefiles module and adds them to
     * the module
     *
     * @param module reference to the module
     * @return 0 on success, -1 on failure
     */
    static int import_exceptions(PyObject* module){
        if (import_exception(module, ERR_NAME("ImanError"),
                "The is the base class for all exceptions in the imageanalysis module", NULL, &PyExc_ImanError,
                NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("IoError"),
                "This is the base class for all exceptions in the imageanalysis.sourcefiles module",
                PyExc_ImanError, &PyExc_ImanIoError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "TrainError", "ihna.kozhukhov.imageanalysis.sourcefiles.TrainError",
                "The is the base class for all I/O exceptions generated within the file train",
                PyExc_ImanIoError, &PyExc_TrainError, "train_name") < 0){
            return -1;
        }
        if (import_exception(module, "ExperimentModeError",
                "ihna.kozhukhov.imageanalysis.sourcefiles.ExperimentModeError",
                "This exception is thrown when you try to read the train property or call the train method that is"
                " absolutely meaningless for the current stimulation protocol. Also, when you try to call this method "
                "before the opening of the train, you will get this exception",
                PyExc_TrainError, &PyExc_ExperimentModeError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "SynchronizationChannelNumberError",
                             FULL_ERROR_NAME_PREFIX"SynchronizationChannelNumberError",
                "The exception is thrown when you try to access the non-existent synchronization channel or "
                "channel that was not used in the experiment",
                PyExc_TrainError, &PyExc_SynchronizationChannelNumberError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "UnsupportedExperimentModeError",
                             FULL_ERROR_NAME_PREFIX"UnsupportedExperimentModeError",
                "The exception is thrown when image analysis have no idea or very confused about what stimulation"
                "protocol is used in your experiment. At the moment of creating an exception trying to read the"
                "file where both COST and EPST chunks were presented or where both of them are absent will"
                "throw this error",
                PyExc_TrainError, &PyExc_UnsupportedExperimentModeError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "SourceFileError",
                             FULL_ERROR_NAME_PREFIX"SourceFileError",
                "The is the base class for all exceptions that are connected to a certain particular file",
                PyExc_TrainError, &PyExc_SourceFileError, "file_name") < 0){
            return -1;
        }
        if (import_exception(module, "FrameNumberError",
                             FULL_ERROR_NAME_PREFIX"FrameNumberError",
                "In order to check the data consistency total number of recorded frames shall be written to the "
                "SOFT chunk. During the load such a number is compared with sum of all frames presented in all files "
                "within the train."
                " If these values don't mismatch to each other this exception will be thrown",
                PyExc_SourceFileError, &PyExc_FrameNumberError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "DataChunkSizeMismatchError",
                             FULL_ERROR_NAME_PREFIX"DataChunkSizeMismatchError",
                "This exception will be thrown is size of the file body is not the same as size of a single frame "
                "multiplied by total number of frames in this file. Maybe, this happens due to the data loss.",
                PyExc_SourceFileError, &PyExc_DataChunkSizeMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "IsoiChunkSizeMismatchError",
                             FULL_ERROR_NAME_PREFIX"IsoiChunkSizeMismatchError",
                "In the valid imaging data file the size of the ISOI chunk shall correspond to the size of "
                "the data chunk plus the size of the file header (footers are not supported by this package, their "
                "presence will definitely throw this error). If this is not true, this exception will be thrown.",
                PyExc_SourceFileError, &PyExc_IsoiChunkSizeMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FileSizeMismatchError",
                             FULL_ERROR_NAME_PREFIX"FileSizeMismatchError",
                "The exception will be thrown during an attempt to read the train from uncompressed (native) "
                "data. In such a data the actual file size detected by the operating system shall be the same"
                "as the size of the ISOI chunk. Failure to do this will raise this error",
                PyExc_SourceFileError, &PyExc_FileSizeMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "ExperimentChunkMismatchError",
                             FULL_ERROR_NAME_PREFIX"ExperimentChunkMismatchError",
                "The stimulation protocol (continuous or episodic) shall be the same during the whole record. Because "
                "such a protocol is defined by so called 'experiment chunk' (either 'COST' or 'EPST') this means "
                "that the same experiment chunk shall be present in all files in the file train",
                PyExc_SourceFileError, &PyExc_ExperimentChunkMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FileHeaderMismatchError",
                             FULL_ERROR_NAME_PREFIX"FileHeaderMismatchError",
                "The program will work correctly when the header size if the same for all files in the chain. "
                "If this is not true this exception will be thrown",
                PyExc_SourceFileError, &PyExc_FileHeaderMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FrameHeaderMismatchError",
                             FULL_ERROR_NAME_PREFIX"FrameHeaderMismatchError",
                "The program will work correctly when all frames in the within the record shall have headers of the "
                "same size. If this is not truth, this error will be thrown",
                PyExc_SourceFileError, &PyExc_FrameHeaderMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FrameDimensionsMismatchError",
                             FULL_ERROR_NAME_PREFIX"FrameDimensionsMismatchError",
                "This error will be thrown if frame dimensions are not constant during the whole record",
                PyExc_SourceFileError, &PyExc_FrameDimensionsMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "DataTypeMismatchError",
                             FULL_ERROR_NAME_PREFIX"DataTypeMismatchError",
                "This error will be thrown is the data type is not the same during the whole record",
                PyExc_SourceFileError, &PyExc_DataTypeMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "CompChunkNotExistError",
                             FULL_ERROR_NAME_PREFIX"CompChunkNotExistError",
                "This error will be thrown on attempt to open the compressed file that doesn't contain the COMP chunk",
                PyExc_SourceFileError, &PyExc_CompChunkNotExistError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FileOpenError",
                             FULL_ERROR_NAME_PREFIX"FileOpenError",
                "This error generates when the operating system fails to open the file for reading",
                PyExc_SourceFileError, &PyExc_FileOpenError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FileReadError",
                             FULL_ERROR_NAME_PREFIX"FileReadError",
                "This error generates when the operating system fails to read the requsted bytes from the file",
                PyExc_SourceFileError, &PyExc_FileReadError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "ChunkError",
                             FULL_ERROR_NAME_PREFIX"ChunkError",
                "The is the base class for all errors related to a certain chunk",
                PyExc_SourceFileError, &PyExc_ChunkError, "chunk_name") < 0){
            return -1;
        }
        if (import_exception(module, "UnsupportedChunkError",
                             FULL_ERROR_NAME_PREFIX"UnsupportedChunkError",
                "This exception will be thrown if the file contains a chunk that can't be recognized by the module "
                "because its type is definitely unknown for the module",
                PyExc_ChunkError, &PyExc_UnsupportedChunkError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "ChunkSizeMismatchError",
                             FULL_ERROR_NAME_PREFIX"ChunkSizeMismatchError",
                "Each chunk type (except ISOI and DATA chunks) is assumed to have the same size for all data you "
                "try to process by this package and saved by this package. Such a size is written inside this module."
                " If you try to read the chunk with non-standard size you will receive this error",
                PyExc_ChunkError, &PyExc_ChunkSizeMismatchError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "ChunkNotFoundError",
                             FULL_ERROR_NAME_PREFIX"ChunkNotFoundError",
                "This error will be thrown when the module requires a certain chunk to be presented within the file "
                "but this chunk is absent in the given data",
                PyExc_ChunkError, &PyExc_ChunkNotFoundError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FileNotOpenedError",
                             FULL_ERROR_NAME_PREFIX"FileNotOpenedError",
                "This error will be thrown when you try to call the method that requires the file to be opened "
                "without necessity to load the file header. Open the file first and then apply this method in order "
                "to fix this error.",
                PyExc_SourceFileError, &PyExc_FileNotOpenedError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "IsoiChunkNotFoundError",
                             FULL_ERROR_NAME_PREFIX"IsoiChunkNotFoundError",
                "This error will be thrown if there is not ISOI chunk presented at the very beginning of the file",
                PyExc_SourceFileError, &PyExc_IsoiChunkNotFoundError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "FileNotLoadedError",
                             FULL_ERROR_NAME_PREFIX"FileNotLoadedError",
                "This error will be raised when you try to call the method that requires file header to be loaded. "
                "Load the file header first, and then apply this method in order to fix this error.",
                PyExc_SourceFileError, &PyExc_FileNotLoadedError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("DataChunkNotFoundError"),
                "This error will be raised when you try to load the file with not DATA chunk presented",
                PyExc_SourceFileError, &PyExc_DataChunkNotFoundError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("NotAnalysisFileError"),
                "This error will be thrown when you try to apply AnalysisSourceFile class for loading the file "
                "that is not an analysis file",
                PyExc_SourceFileError, &PyExc_NotAnalysisFileError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("NotGreenFileError"),
                "This error will be thrown when you try to apply GreenSourceFile class for loading the file "
                "that is not a green file",
                             PyExc_SourceFileError, &PyExc_NotGreenFileError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("NotTrainHeadError"),
                "This error will be thrown when the loading file is not the head of the file train and you  asked "
                "to throw an exception when you adjusted parameters of the TrainSourceFile or any of its descendants",
                PyExc_SourceFileError, &PyExc_NotTrainHeadError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("NotStreamFileError"),
                "This error will be thrown when you apply StreamSourceFile class for loading the file that is "
                "not a stream file or present in the compressed mode",
                PyExc_SourceFileError, &PyExc_NotStreamFileError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, ERR_NAME("NotCompressedFileError"),
                "This error will be thrown when you try to apply CompressedSourceFile to load the file the is not "
                "compressed or doesn't contain a native imaging signal",
                PyExc_SourceFileError, &PyExc_NotCompressedFileError, NULL) < 0){
            return -1;
        }

        return 0;
    }

    static void clear_all_exceptions(void){
        Py_XDECREF(PyExc_ImanError);
        Py_XDECREF(PyExc_ImanIoError);
        Py_XDECREF(PyExc_TrainError);
        Py_XDECREF(PyExc_ExperimentModeError);
        Py_XDECREF(PyExc_SynchronizationChannelNumberError);
        Py_XDECREF(PyExc_SourceFileError);
        Py_XDECREF(PyExc_FrameNumberError);
        Py_XDECREF(PyExc_DataChunkSizeMismatchError);
        Py_XDECREF(PyExc_DataChunkSizeMismatchError);
        Py_XDECREF(PyExc_IsoiChunkSizeMismatchError);
        Py_XDECREF(PyExc_FileSizeMismatchError);
        Py_XDECREF(PyExc_FileSizeMismatchError);
        Py_XDECREF(PyExc_ExperimentChunkMismatchError);
        Py_XDECREF(PyExc_FileHeaderMismatchError);
        Py_XDECREF(PyExc_FrameHeaderMismatchError);
        Py_XDECREF(PyExc_FrameDimensionsMismatchError);
        Py_XDECREF(PyExc_DataTypeMismatchError);
        Py_XDECREF(PyExc_CompChunkNotExistError);
        Py_XDECREF(PyExc_FileOpenError);
        Py_XDECREF(PyExc_ChunkError);
        Py_XDECREF(PyExc_UnsupportedChunkError);
        Py_XDECREF(PyExc_ChunkSizeMismatchError);
        Py_XDECREF(PyExc_ChunkNotFoundError);
        Py_XDECREF(PyExc_FileNotOpenedError);
        Py_XDECREF(PyExc_IsoiChunkNotFoundError);
        Py_XDECREF(PyExc_FileNotLoadedError);
        Py_XDECREF(PyExc_DataChunkNotFoundError);
        Py_XDECREF(PyExc_NotAnalysisFileError);
        Py_XDECREF(PyExc_NotGreenFileError);
        Py_XDECREF(PyExc_NotTrainHeadError);
        Py_XDECREF(PyExc_NotStreamFileError);
        Py_XDECREF(PyExc_NotCompressedFileError);
    }

    PyMODINIT_FUNC PyInit__exceptions(void){
        PyObject* module;
        static void* _exceptions_API[C_API_INSTANCE_NUMBER];
        PyObject* _exceptions_capsule = NULL;


        module = PyModule_Create(&core);
        if (module == NULL){
            printf("Error in creating the module\n");
            return NULL;
        }
        if (import_exceptions(module) != 0){
            clear_all_exceptions();
            Py_DECREF(module);
            return NULL;
        }

        _exceptions_API[C_API_Exception_process] = (void*)Exception_process;
        _exceptions_API[C_API_ImanError] = PyExc_ImanError;
        _exceptions_API[C_API_IoError] = PyExc_ImanIoError;
        _exceptions_API[C_API_TrainError] = PyExc_TrainError;
        _exceptions_API[C_API_ExperimentModeError] = PyExc_ExperimentModeError;
        _exceptions_API[C_API_SynchronizationChannelNumberError] = PyExc_SynchronizationChannelNumberError;
        _exceptions_API[C_API_UnsupportedExperimentModeError] = PyExc_UnsupportedExperimentModeError;
        _exceptions_API[C_API_SourceFileError] = PyExc_SourceFileError;
        _exceptions_API[C_API_FrameNumberError] = PyExc_FrameNumberError;
        _exceptions_API[C_API_DataChunkNotFoundError] = PyExc_DataChunkNotFoundError;
        _exceptions_API[C_API_IsoiChunkSizeMismatchError] = PyExc_IsoiChunkSizeMismatchError;
        _exceptions_API[C_API_FileSizeMismatchError] = PyExc_FileSizeMismatchError;
        _exceptions_API[C_API_ExperimentChunkMismatchError] = PyExc_ExperimentChunkMismatchError;
        _exceptions_API[C_API_FileHeaderMismatchError] = PyExc_FileHeaderMismatchError;
        _exceptions_API[C_API_FrameHeaderMismatchError] = PyExc_FrameHeaderMismatchError;
        _exceptions_API[C_API_FrameDimensionsMismatchError] = PyExc_FrameDimensionsMismatchError;
        _exceptions_API[C_API_DataTypeMismatchError] = PyExc_DataTypeMismatchError;
        _exceptions_API[C_API_CompChunkNotExistError] = PyExc_CompChunkNotExistError;
        _exceptions_API[C_API_FileNotOpenedError] = PyExc_FileNotOpenedError;
        _exceptions_API[C_API_FileOpenError] = PyExc_FileOpenError;
        _exceptions_API[C_API_FileReadError] = PyExc_FileReadError;
        _exceptions_API[C_API_ChunkError] = PyExc_ChunkError;
        _exceptions_API[C_API_UnsupportedChunkError] = PyExc_UnsupportedChunkError;
        _exceptions_API[C_API_ChunkSizeMismatchError] = PyExc_ChunkSizeMismatchError;
        _exceptions_API[C_API_ChunkNotFoundError] = PyExc_ChunkNotFoundError;
        _exceptions_API[C_API_IsoiChunkNotFoundError] = PyExc_IsoiChunkNotFoundError;
        _exceptions_API[C_API_FileNotLoadedError] = PyExc_FileNotLoadedError;
        _exceptions_API[C_API_DataChunkSizeMismatchError] = PyExc_DataChunkSizeMismatchError;
        _exceptions_API[C_API_FileNotLoadedError] = PyExc_FileNotLoadedError;
        _exceptions_API[C_API_NotAnalysisFileError] = PyExc_NotAnalysisFileError;
        _exceptions_API[C_API_NotGreenFileError] = PyExc_NotGreenFileError;
        _exceptions_API[C_API_NotCompressedFileError] = PyExc_NotCompressedFileError;
        _exceptions_API[C_API_NotStreamFileError] = PyExc_NotStreamFileError;
        _exceptions_API[C_API_NotTrainHeadError] = PyExc_NotTrainHeadError;

        _exceptions_capsule = PyCapsule_New(_exceptions_API, FULL_ERROR_NAME_PREFIX"_exceptions._c_API", NULL);
        if (_exceptions_capsule == NULL){
            clear_all_exceptions();
            Py_DECREF(module);
            return NULL;
        }
        if (PyModule_AddObject(module, "_c_API", _exceptions_capsule) < 0){
            Py_DECREF(_exceptions_capsule);
            clear_all_exceptions();
            Py_DECREF(module);
            return NULL;
        }

        return module;
    }

}
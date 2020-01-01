//
// Created by serik1987 on 01.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAME_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAME_H

#include "../exceptions.h"
#include "FileTrain.h"
#include "FramChunk.h"
#include "FramEpstChunk.h"
#include "FramCostChunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * The class represents the data connected to a single frame
     */
    class Frame {
        const FileTrain* parent;
        int frameNumber = 0;
        FramChunk* framChunk;
        Chunk* stimulationChunk;
        FileTrain::ExperimentalMode emode;
        uint16_t* body = nullptr;

    public:

        class frame_exception: public FileTrain::train_exception{
        private:
            int frameNumber;
            std::string message;

        public:
            frame_exception(const FileTrain* train, int n, const std::string& msg):
                train_exception(msg, train), frameNumber(n), message(train->getFilename() + " [ frame # " +
                    std::to_string(frameNumber) + " ] " + msg) {};

            frame_exception(const std::string& train_name, int n, const std::string& msg):
                train_exception(msg, train_name), frameNumber(n) {};

            [[nodiscard]] int getFrameNumber() const { return frameNumber; }

            [[nodiscard]] const char* what() const noexcept override { return message.c_str(); }
        };

        class frame_not_read: public frame_exception{
        public:
            frame_not_read(const FileTrain* train, int n):
                frame_exception(train, n, MSG_FRAME_NOT_READ) {};

            frame_not_read(const std::string& train_name, int n):
                frame_exception(train_name, n, MSG_FRAME_NOT_READ) {};
        };

        class frame_is_out_of_range: public frame_exception{
        public:
            explicit frame_is_out_of_range(const FileTrain* train, int n):
                    frame_exception(train, n, MSG_FRAME_OUT_OF_RANGE) {};
            explicit frame_is_out_of_range(const std::string& trainname, int n):
                    frame_exception(trainname, n, MSG_FRAME_OUT_OF_RANGE) {};
        };


        /**
         * The frame initialization
         *
         * @param train reference to the file train
         */
        explicit Frame(const FileTrain& train);

        Frame(const Frame& other) = delete;
        Frame& operator=(const Frame& other) = delete;

        ~Frame();

        /**
         *
         * @return the frame number
         */
        [[nodiscard]] int getFrameNumber() const { return frameNumber; }

        /**
         *
         * @return the content of the FRAM chunk. If the frame is not read, throws frame_not_read exception
         */
        FramChunk& getFramChunk() {
            if (framChunk == nullptr){
                throw frame_not_read(parent, frameNumber);
            } else {
                return *framChunk;
            }
        }

        /**
         *
         * @return pointer to the experiment chunk or nullptr if the frame has not been loaded
         */
        Chunk* getExperimentChunk() {
            return stimulationChunk;
        }

        /**
         *
         * @return cost chunk from the frame header. If the frame has not been read, the function throws
         * frame_not_read. If no cost chunk has been present, the function throws FileTrain::experiment_mode_exception
         */
        FramCostChunk& getCostChunk() {
            if (stimulationChunk == nullptr){
                throw frame_not_read(parent, frameNumber);
            }
            auto* chunk = dynamic_cast<FramCostChunk*>(stimulationChunk);
            if (chunk == nullptr){
                throw FileTrain::experiment_mode_exception(parent);
            }
            return *chunk;
        }

        /**
         *
         * @return the reference to the frame epst chunk. If the frame has not been readm the function throws
         * frame_not_read. If no epst chunk has been present, the function throws FileTrain::experiment_mode_exception
         */
        FramEpstChunk& getEpstChunk() {
            if (stimulationChunk == nullptr){
                throw frame_not_read(parent, frameNumber);
            }
            auto* chunk = dynamic_cast<FramEpstChunk*>(stimulationChunk);
            if (chunk == nullptr){
                throw FileTrain::experiment_mode_exception(parent);
            }
            return *chunk;
        }

        /**
         * If this public variable is true the frame is considered to be LOCKED. Its delete from the operating
         * memory will not be possible unless the train will be deleted
         */
        bool iLock;

        friend std::ostream& operator<<(std::ostream& out, const Frame& frame);


    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAME_H

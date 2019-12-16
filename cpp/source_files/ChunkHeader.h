//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNKHEADER_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNKHEADER_H

#include <cstdint>
#include <string>
#include <cstring>
#include "../core.h"

#include "FramChunk.h"
#include "FramCostChunk.h"
#include "FramEpstChunk.h"
#include "HardChunk.h"
#include "SoftChunk.h"
#include "CostChunk.h"
#include "EpstChunk.h"
#include "GreenChunk.h"
#include "SyncChunk.h"
#include "RoisChunk.h"
#include "CompChunk.h"

namespace iman {

    /**
     * This is the base class to represent the chunk header and create the chunk based on
     * the chunk header
     *
     * See Chunk help for details
     */
    class ChunkHeader {
    public:
        static constexpr int CHUNK_ID_SIZE = 4;
        static constexpr int CHUNK_TAG_SIZE = 4;

#pragma PACK(push, 1)
        struct DATA_CHUNK{
            char ID[CHUNK_ID_SIZE];
            uint32_t size;
        };
#pragma PACK(pop)

        static constexpr uint32_t FRAM_CHUNK_CODE = 1296126534;
        static constexpr uint32_t cost_CHUNK_CODE = 1953722211;
        static constexpr uint32_t ISOI_CHUNK_CODE = 1229935433;
        static constexpr uint32_t SOFT_CHUNK_CODE = 1413893971;
        static constexpr uint32_t DATA_CHUNK_CODE = 1096040772;
        static constexpr uint32_t COST_CHUNK_CODE = 1414745923;
        static constexpr uint32_t COMP_CHUNK_CODE = 1347243843;
        static constexpr uint32_t HARD_CHUNK_CODE = 1146241352;
        static constexpr uint32_t ROIS_CHUNK_CODE = 1397313362;
        static constexpr uint32_t SYNC_CHUNK_CODE = 1129208147;
        static constexpr uint32_t epst_CHUNK_CODE = 1953722469;
        static constexpr uint32_t EPST_CHUNK_CODE = 1414746181;
        static constexpr uint32_t GREE_CHUNK_CODE = 1162170951;
        static constexpr uint32_t INVALID_CHUNK_CODE = 4294967295;

    private:
        DATA_CHUNK header;

        static constexpr int CHUNK_CODE_NUMBER = 14;

        static constexpr uint32_t CHUNK_CODE_LIST[] = {
                FRAM_CHUNK_CODE,
                cost_CHUNK_CODE,
                ISOI_CHUNK_CODE,
                SOFT_CHUNK_CODE,
                DATA_CHUNK_CODE,
                COST_CHUNK_CODE,
                COMP_CHUNK_CODE,
                HARD_CHUNK_CODE,
                ROIS_CHUNK_CODE,
                SYNC_CHUNK_CODE,
                epst_CHUNK_CODE,
                EPST_CHUNK_CODE,
                GREE_CHUNK_CODE,
                INVALID_CHUNK_CODE
        };

        static constexpr uint32_t CHUNK_SIZE_LIST[] = {
                sizeof(FramChunk::FRAM_CHUNK),
                sizeof(FramCostChunk::FRAM_COST_CHUNK),
                0,
                sizeof(SoftChunk::SOFT_CHUNK),
                0,
                sizeof(CostChunk::COST_CHUNK),
                sizeof(CompChunk::COMP_CHUNK),
                sizeof(HardChunk::HARD_CHUNK),
                sizeof(RoisChunk::ROIS_CHUNK),
                sizeof(SyncChunk::SYNC_CHUNK),
                sizeof(FramEpstChunk::FRAM_EPST_CHUNK),
                sizeof(EpstChunk::EPST_CHUNK),
                sizeof(GreenChunk::GREE_CHUNK),
                0
        };

    public:

        /**
         * Creates new chunk header
         *
         * @param dataChunk structure of the chunk header
         */
        explicit ChunkHeader(DATA_CHUNK& dataChunk): header(dataChunk) {};

        /**
         * Creates new chunk header
         *
         * @param name chunk name
         * @param size chunk size
         */
        ChunkHeader(const char* name, uint32_t size): header() {
            std::strncpy(header.ID, name, sizeof(header.ID));
            header.size = size;
        }

        /**
         *
         * @return the chunk header ID
         */
        [[nodiscard]] std::string getChunkId() const{
            return std::string(header.ID, CHUNK_ID_SIZE);
        }

        /**
         *
         * @return the chunk ID as a non-terminated string
         */
        [[nodiscard]] const char* getChunkIdRaw() const{
            return header.ID;
        }

        /**
         *
         * @return size of the chunk body in bytes
         */
        [[nodiscard]] uint32_t getChunkSize() const{
            return header.size;
        }

        /**
         * Checks whether the chunk has an appropriate ID
         *
         * @param id chunk ID to check
         * @return true if the chunk ID is the same as given ID
         */
        bool operator==(const char* id) const{
            return std::strncmp(header.ID, id, CHUNK_ID_SIZE) == 0;
        }

        /**
         * Checks whether the chunk has an appropriate ID
         *
         * @param id chunk ID to check
         * @return true if the chunk ID is the same as given ID
         */
        bool operator==(const std::string& id) const{
            return operator==(id.c_str());
        }

        /**
         * Compares two chunks
         * Two chunks are treated to be equal to each other if
         * and only if their IDs are the same
         *
         * @param other the other chunk to compare
         * @return true if two chunks are equal
         */
        bool operator==(const ChunkHeader& other) const{
            return operator==(other.header.ID);
        }

        /**
         * Checks whether the chunk has an appropriate code
         *
         * @param chunk_code The chunk code to compare
         * @return true if the chunk code is the same as a given argument
         */
        bool operator==(int chunk_code) const{
            return operator uint32_t() == chunk_code;
        }

        /**
         * Checks whether the chunk has an appropriate ID
         *
         * @param id the chunk ID
         * @return true if the chunk doesn't contain an appropriate ID
         */
        bool operator!=(const char* id) const{
            return strncmp(header.ID, id, CHUNK_ID_SIZE) != 0;
        }

        /**
         * Checks whether the chunk has an appropriate ID
         *
         * @param id the chunk ID
         * @return true if the chunk doesn't contain an appropriate ID
         */
        bool operator!=(const std::string& id) const{
            return operator!=(id.c_str());
        }

        /**
         * Checks whether two chunks are not equal to each other
         * Two chunks are not equal to each other if and only if
         * their IDs are not the same
         *
         * @param other the other chunk to compare
         * @return true if two chunks are not equal to each other
         */
        bool operator!=(const ChunkHeader& other) const{
            return operator!=(other.header.ID);
        }

        /**
         * Checks whether the chunk has an appropriate code
         *
         * @param chunk_code The chunk code to compare
         * @return true if the chunk code is not the same as a given argument
         */
        bool operator!=(int chunk_code) const{
            return operator uint32_t() != chunk_code;
        }

        /**
         *
         * @return the so called "chunk code" Chunk code is a chunk header transformed to integer
         */
        explicit operator uint32_t() const{
            return *((int*)header.ID);
        }

        /**
         * Checks for chunk support
         * The chunk is treated to be supported if this is within the list of known chunks
         *
         * @return true if the chunk is known
         */
        [[nodiscard]] bool isKnown() const;

        /**
         * Checks for chunk support
         * The chunk is treated to be supported if this is within the list of known chunks
         *
         * @param id ID of the chunk to check
         * @return true if the chunk is known
         */
        [[nodiscard]] static bool isKnown(const char* id);

        /**
         * Checks for chunk support
         * The chunk is treated to be supported if this is within the list of known chunks
         *
         * @param s ID of the chunk to check
         * @return true if the chunk is known
         */
        [[nodiscard]] static bool isKnown(const std::string& s) { return isKnown(s.c_str()); }

        /**
         *
         * @return true if the chunk is not valid
         */
        [[nodiscard]] bool isValid() {
            return operator uint32_t() != INVALID_CHUNK_CODE;
        }

        /**
         *
         * @return true if the chunk is not valid
         */
        [[nodiscard]] bool isInvalid(){
            return operator uint32_t() == INVALID_CHUNK_CODE;
        }

        /**
         * Shall be thrown when the chunk name presented in the input argument is unknown or invalid
         */
        class unsupported_chunk_exception: public iman_exception{
        public:
            explicit unsupported_chunk_exception(const char* chunk):
                iman_exception("Unknown or unsupported chunk: " + std::string(chunk, CHUNK_ID_SIZE)) {};
        };

        /**
         * Throws when the actual chunk size defined by the Size field in the chunk header is not the same
         * as the desired chunk size defined total size of all data placed into this chunk
         */
        class chunk_size_mismatch_exception: public iman_exception{
        public:
            explicit chunk_size_mismatch_exception(const char* chunkName):
                iman_exception("Incorrect or corrupter chunk " + std::string(chunkName, CHUNK_ID_SIZE)) {};
        };

        static ChunkHeader InvalidChunk() { return ChunkHeader("\xFF\xFF\xFF\xFF", 0); }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNKHEADER_H

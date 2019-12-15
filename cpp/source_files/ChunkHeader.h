//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNKHEADER_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNKHEADER_H

#include <cstdint>
#include <string>
#include <cstring>
#include "../core.h"

namespace iman {

    /**
     * This is the base class to represent the chunk header and create the chunk based on
     * the chunk header
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

        static constexpr int FRAM_CHUNK_CODE = 1296126534;
        static constexpr int cost_CHUNK_CODE = 1953722211;
        static constexpr int ISOI_CHUNK_CODE = 1229935433;
        static constexpr int SOFT_CHUNK_CODE = 1413893971;
        static constexpr int DATA_CHUNK_CODE = 1096040772;
        static constexpr int COST_CHUNK_CODE = 1414745923;
        static constexpr int COMP_CHUNK_CODE = 1347243843;
        static constexpr int HARD_CHUNK_CODE = 1146241352;
        static constexpr int ROIS_CHUNK_CODE = 1397313362;
        static constexpr int SYNC_CHUNK_CODE = 1129208147;
        static constexpr int epst_CHUNK_CODE = 1953722469;
        static constexpr int EPST_CHUNK_CODE = 1414746181;
        static constexpr int GREE_CHUNK_CODE = 1162170951;
        static constexpr int INVALID_CHUNK_CODE = -1;

    private:
        DATA_CHUNK header;

        static constexpr int CHUNK_CODE_NUMBER = 14;

        static constexpr int CHUNK_CODE_LIST[] = {
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
         * @return chunk size in bytes
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
            return operator int() == chunk_code;
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
            return operator int() != chunk_code;
        }

        /**
         *
         * @return the so called "chunk code" Chunk code is a chunk header transformed to integer
         */
        explicit operator int() const{
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
            return operator int() != INVALID_CHUNK_CODE;
        }

        /**
         *
         * @return true if the chunk is not valid
         */
        [[nodiscard]] bool isInvalid(){
            return operator int() == INVALID_CHUNK_CODE;
        }

        /**
         * Shall be thrown when the chunk name presented in the input argument is unknown or invalid
         */
        class unsupported_chunk_exception: public iman_exception{
        public:
            explicit unsupported_chunk_exception(const char* chunk):
                iman_exception("Unknown or unsupported chunk: " + std::string(chunk, CHUNK_ID_SIZE)) {};
        };

        static ChunkHeader InvalidChunk() { return ChunkHeader("\xFF\xFF\xFF\xFF", 0); }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNKHEADER_H

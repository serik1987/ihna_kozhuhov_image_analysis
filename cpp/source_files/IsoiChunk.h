//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_ISOICHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_ISOICHUNK_H

#include <list>
#include <unordered_map>
#include "Chunk.h"
#include "SoftChunk.h"
#include "DataChunk.h"

namespace ihna::kozhukhov::image_analysis {

    /**
     * This is the Main Container Chunk that contains all other chunks
     * presented in the file.
     *
     * The size of the ISOI chunk is the same as the file size. However, when the file
     * is compressed, actual size of ISOI chunk is not the same as value defined in the
     * header of this chunk, such value reflects the file size in the uncompressed state
     */
    class IsoiChunk: public Chunk {
    private:
        static constexpr int ISOI_CHUNK_TAG_SIZE = 4;

        char Tag[ISOI_CHUNK_TAG_SIZE] = "\x00\x00\x00";
        std::list<Chunk*> header_chunks;
        std::unordered_map<uint32_t, Chunk*> chunk_map;
        SoftChunk* softChunk = nullptr;
        DataChunk* dataChunk = nullptr;
    public:
        explicit IsoiChunk(uint32_t size): Chunk("ISOI", size){
            body = nullptr;
        };

        ~IsoiChunk();

        /**
         *
         * @return the ISOI chunk tag transformed into char
         */
        [[nodiscard]] std::string getTag() const {
            return std::string(Tag, ISOI_CHUNK_TAG_SIZE);
        };

        /**
         * Reads the whole file header and the whole file footer from the file.
         * After finishing this function the file pointer is guaranteed to be at the beginning
         * of the file data (the body of CHUNK data
         *
         * @param file file which header and footer shall be read
         */
        void readFromFile(SourceFile& file) override;

        /**
         *
         * @return iterator to the beginning of the header chunks
         */
        [[nodiscard]] std::list<Chunk*>::iterator hbegin() { return header_chunks.begin(); }

        /**
         *
         * @return constant iterator to the beginning of the header chunks
         */
        [[nodiscard]] std::list<Chunk*>::const_iterator chbegin() const { return header_chunks.cbegin(); }

        /**
         *
         * @return iterator to the end of the header chunks
         */
        std::list<Chunk*>::iterator hend() { return header_chunks.end(); }

        /**
         *
         * @return constant iterator to the end of the header chunks
         */
        [[nodiscard]] std::list<Chunk*>::const_iterator chend() const { return header_chunks.cend(); }

        /**
         * Finds the chunk containing in ISOI chunk.
         * If there are several chunks containing in the file, the function returns just one of them
         * picked up randomly
         * The function doesn't find chunks containin in the frame
         * Trying to find the ISOI chunk will return the pointer to the calluing object
         *
         * @param chunkId the chunk code (some constant from ChunkHeader namespace ending by _CODE)
         * @return pointer to the chunk found or nullptr if such chunk doesn't exist
         */
        Chunk* getChunkById(uint32_t chunkId);

        /**
         * Throws text information about the ISOI chunk into the output stream
         *
         * @param out the stream that receives such an information
         * @param isoi reference to the ISOI chunk
         * @return reference to out
         */
        friend std::ostream& operator<<(std::ostream& out, const IsoiChunk& isoi);

        /**
         * The class is suitable for iteration through the whole structure of the file. The class
         * imlements a starndard bi-directional iterator
         */
        class iterator{
        public:
            enum ChunkLocation {InSoft, InHeader, InBody, NoChunk};

        private:
            ChunkLocation location;
            std::list<Chunk*>::iterator it;
            Chunk* referred_object = nullptr;
            IsoiChunk* isoiChunk = nullptr;

        public:
            iterator(ChunkLocation loc, const std::list<Chunk*>::iterator& it,
                    Chunk* pointer = nullptr, IsoiChunk* parent = nullptr):
                location(loc), it(it), referred_object(nullptr), isoiChunk(parent) {};

            bool operator==(const iterator& other) const {
                if (location != other.location) return false;
                if (location == InSoft || location == NoChunk || location == InBody) return true;
                return it == other.it;
            }

            bool operator!=(const iterator& other) const {
                if (location != other.location) return true;
                if (location == InSoft || location == NoChunk || location == InBody) return false;
                return it != other.it;
            }

            Chunk& operator*(){
                if (referred_object == nullptr){
                    if (location == InSoft){
                        referred_object = isoiChunk->softChunk;
                    }
                    if (location == InHeader){
                        referred_object = *it;
                    }
                    if (location == InBody){
                        referred_object = isoiChunk->dataChunk;
                    }
                    if (location == NoChunk){
                        throw std::bad_alloc();
                    }
                }

                return *referred_object;
            }

            Chunk* operator->(){
                return &operator*();
            }

            iterator& operator++(){
                if (location == InSoft){
                    location = InHeader;
                    it = isoiChunk->hbegin();
                    referred_object = *it;
                    return *this;
                }
                if (location == InHeader){
                    ++it;
                    if (it == isoiChunk->hend()){
                        location = InBody;
                        referred_object = isoiChunk->dataChunk;
                    } else {
                        referred_object = *it;
                    }
                    return *this;
                }
                if (location == InBody){
                    location = NoChunk;
                    referred_object = nullptr;
                    return *this;
                }

                return *this;
            }

            iterator operator++(int){
                iterator copy = *this;
                ++*this;
                return copy;
            }

            iterator& operator--(){
                if (location == InSoft){
                    return *this;
                }
                if (location == InHeader){
                    if (it == isoiChunk->hbegin()){
                        location = InSoft;
                        it = isoiChunk->hend();
                        referred_object = isoiChunk->softChunk;
                        return *this;
                    } else {
                        --it;
                        referred_object = *it;
                        return *this;
                    }
                }
                if (location == InBody){
                    location = InHeader;
                    it = isoiChunk->hend();
                    --it;
                    referred_object = *it;
                    return *this;
                }
                if (location == NoChunk){
                    location = InBody;
                    referred_object = isoiChunk->dataChunk;
                    return *this;
                }

                return *this;
            }

            iterator operator--(int){
                iterator temp = *this;
                --*this;
                return temp;
            }
        };

        [[nodiscard]] iterator begin() {
            if (softChunk == nullptr){
                return end();
            } else {
                return iterator(iterator::InSoft, hend(), (Chunk *) softChunk, this);
            }
        }

        [[nodiscard]] iterator end() {
            return iterator(iterator::NoChunk, hend(), nullptr, this);
        }

        /**
         * The class is suitable for iteration through the whole structure of the file. The class
         * imlements a starndard bi-directional iterator
         */
        class const_iterator{
        public:
            enum ChunkLocation {InSoft, InHeader, InBody, NoChunk};

        private:
            ChunkLocation location;
            std::list<Chunk*>::const_iterator it;
            const Chunk* referred_object = nullptr;
            const IsoiChunk* isoiChunk = nullptr;

        public:
            const_iterator(ChunkLocation loc, const std::list<Chunk*>::const_iterator& it,
                    const Chunk* pointer = nullptr, const IsoiChunk* parent = nullptr):
                    location(loc), it(it), referred_object(nullptr), isoiChunk(parent) {};

            bool operator==(const const_iterator& other) const {
                if (location != other.location) return false;
                if (location == InSoft || location == NoChunk || location == InBody) return true;
                return it == other.it;
            }

            bool operator!=(const const_iterator& other) const {
                if (location != other.location) return true;
                if (location == InSoft || location == NoChunk || location == InBody) return false;
                return it != other.it;
            }

            const Chunk& operator*(){
                if (referred_object == nullptr){
                    if (location == InSoft){
                        referred_object = isoiChunk->softChunk;
                    }
                    if (location == InHeader){
                        referred_object = *it;
                    }
                    if (location == InBody){
                        referred_object = isoiChunk->dataChunk;
                    }
                    if (location == NoChunk){
                        throw std::bad_alloc();
                    }
                }

                return *referred_object;
            }

            const Chunk* operator->(){
                return &operator*();
            }

            const_iterator& operator++(){
                if (location == InSoft){
                    location = InHeader;
                    it = isoiChunk->chbegin();
                    referred_object = *it;
                    return *this;
                }
                if (location == InHeader){
                    ++it;
                    if (it == isoiChunk->chend()){
                        location = InBody;
                        referred_object = isoiChunk->dataChunk;
                    } else {
                        referred_object = *it;
                    }
                    return *this;
                }
                if (location == InBody){
                    location = NoChunk;
                    referred_object = nullptr;
                    return *this;
                }

                return *this;
            }

            const_iterator operator++(int){
                const_iterator copy = *this;
                ++*this;
                return copy;
            }

            const_iterator& operator--(){
                if (location == InSoft){
                    return *this;
                }
                if (location == InHeader){
                    if (it == isoiChunk->chbegin()){
                        location = InSoft;
                        it = isoiChunk->chend();
                        referred_object = isoiChunk->softChunk;
                        return *this;
                    } else {
                        --it;
                        referred_object = *it;
                        return *this;
                    }
                }
                if (location == InBody){
                    location = InHeader;
                    it = isoiChunk->chend();
                    --it;
                    referred_object = *it;
                    return *this;
                }
                if (location == NoChunk){
                    location = InBody;
                    referred_object = isoiChunk->dataChunk;
                    return *this;
                }

                return *this;
            }

            const_iterator operator--(int){
                const_iterator temp = *this;
                --*this;
                return temp;
            }
        };

        /**
         *
         * @return constant iterator to the end of the chunk sequence
         */
        const_iterator cend() const {
            return const_iterator(const_iterator::NoChunk, chend(), nullptr, this);
        }

        const_iterator cbegin() const {
            if (softChunk == nullptr){
                return cend();
            } else {
                return const_iterator(const_iterator::InSoft, chend(), (const Chunk*) softChunk, this);
            }
        }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_ISOICHUNK_H

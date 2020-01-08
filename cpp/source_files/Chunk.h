//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNK_H

#include "../../init.h"

#include <string>
#include <iostream>
#include "SourceFile.h"

namespace GLOBAL_NAMESPACE {

    class SourceFile;

    /**
     * This is a base class for all chunks
     *
     * All data stored in the IMAN source file were grouped into several
     * sections, or "chunks". The chunk contains all data related to a certain aspect of
     * the experimental data acquisition. Each chunk has a separated and stereotyped beginning
     * The chunk has name which reflects the chunk type, or approximate meaning of the data stored
     * in the chunk. For example,
     *
     * FRAM chunk contains general information about a certain frame
     * cost chunk contains information about the frame that was acquired due to continuous stimulation
     * epst chunk contains information about the frame that was acquired due to episodic stimulation
     * SOFT chunk contains general information about the experiment
     * COST chunk contains general information about the continuous stimulation
     * EPST chunk contains general information about the episodic stimulation
     * DATA chunk contains certain content of all frames
     *
     * Each chunk consists of chunk header and chunk body. The chunk header contains information
     * about chunk name and size of the chunk body. The chunk body contains data related to the experiment
     */
    class Chunk {
    protected:
        std::string name;
        uint32_t chunkSize;
        char* body = nullptr;

        virtual void writeBody(std::ofstream& out);

    public:
        /**
         * Initializes the chunk.
         * Puts the chunk name, chunk size and fills the buffer body that will be substituted
         * into ifstream::read, ofstream::write (doesn't fill the base constructor, fill it in the
         * derived one)
         *
         * @param chunk_name all parameters reflects the chunk header
         * @param chunk_size
         */
        Chunk(std::string chunk_name, uint32_t chunk_size): name(std::move(chunk_name)), chunkSize(chunk_size) {};

        virtual ~Chunk(){
#ifdef DEBUG_DELETE_CHECK
            std::cout << "DELETE THE CHUNK\n";
#endif
        }

        /**
         *
         * @return chunk name (non-changing constant that depends on a certain file
         */
        [[nodiscard]] const std::string& getName() const { return name; }

        /**
         *
         * @return size of the chunk body
         */
        [[nodiscard]] virtual uint32_t getSize() const { return chunkSize; }

        /**
         * Reads the string from the string buffer (i.e., from a field like char[n]
         * Transforms the non-NULL-terminated string buffer which size doesn't exceed maxSize to the C++ std::string
         *
         * @param buffer the string buffer
         * @param maxSize max size
         * @return transformation results
         */
        static std::string readString(const char* buffer, int maxSize){
            std::string pre_s(buffer, maxSize);
            auto pos = pre_s.find('\x00');
            auto s = pre_s.substr(0, pos);
            return s;
        }

        /**
         * Reads the chunk from the source file. Your current position within the file is assumed to be
         * at the end of the chunk header. After the reading will be finished, the file position will be moved
         * to the end.
         * The function is overriden in the DATA and ISOI chunks. See help on these chunks for details
         *
         * @param file reference to the SourceFile& from which the chunk shall be read
         */
        virtual void readFromFile(SourceFile& file);

        /**
         * Prints text information about the chunk to the output stream.
         *
         * @param out the output stream
         * @param chunk reference to the chunk
         * @return reference to the out
         */
        friend std::ostream& operator<<(std::ostream& out, const Chunk& chunk);

        /**
         * Write the chunk to the output stream
         *
         * @param filename - name to be substituted to an exception if this is generated
         * @param output the output stream
         */
        void write(const std::string& filename, std::ofstream& output);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_CHUNK_H

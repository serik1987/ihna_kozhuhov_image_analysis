//
// Created by serik1987 on 15.12.2019.
//

#include "ChunkHeader.h"

namespace iman{

    bool ChunkHeader::isKnown() const {
        int chunk_code = operator int();

        for (int i=0; i < CHUNK_CODE_NUMBER; ++i){
            if (CHUNK_CODE_LIST[i] == chunk_code){
                return true;
            }
        }

        return false;
    }

    bool ChunkHeader::isKnown(const char *id) {
        int chunk_code = *(const int*)id;

        for (int i=0; i < CHUNK_CODE_NUMBER; ++i){
            if (CHUNK_CODE_LIST[i] == chunk_code){
                return true;
            }
        }

        return false;
    }
}

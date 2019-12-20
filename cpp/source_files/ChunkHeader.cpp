//
// Created by serik1987 on 15.12.2019.
//

#include "ChunkHeader.h"

#include "IsoiChunk.h"
#include "DataChunk.h"
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

namespace GLOBAL_NAMESPACE{

    const uint32_t ChunkHeader::CHUNK_SIZE_LIST[] = {
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

    bool ChunkHeader::isKnown() const {
        int chunk_code = operator uint32_t();

        for (int i=0; i < CHUNK_CODE_NUMBER; ++i){
            if (CHUNK_CODE_LIST[i] == chunk_code){
                if (CHUNK_SIZE_LIST[i] > 0){
                    if (CHUNK_SIZE_LIST[i] != getChunkSize()){
                        throw chunk_size_mismatch_exception(getChunkIdRaw());
                    }
                }
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

    Chunk *ChunkHeader::createChunk() const {
        Chunk* chunk = nullptr;

        switch (operator uint32_t()){
            case FRAM_CHUNK_CODE:
                chunk = new FramChunk(getChunkSize());
                break;
            case cost_CHUNK_CODE:
                chunk = new FramCostChunk(getChunkSize());
                break;
            case epst_CHUNK_CODE:
                chunk = new FramEpstChunk(getChunkSize());
                break;
            case ISOI_CHUNK_CODE:
                chunk = new IsoiChunk(getChunkSize());
                break;
            case HARD_CHUNK_CODE:
                chunk = new HardChunk(getChunkSize());
                break;
            case SOFT_CHUNK_CODE:
                chunk = new SoftChunk(getChunkSize());
                break;
            case EPST_CHUNK_CODE:
                chunk = new EpstChunk(getChunkSize());
                break;
            case GREE_CHUNK_CODE:
                chunk = new GreenChunk(getChunkSize());
                break;
            case DATA_CHUNK_CODE:
                chunk = new DataChunk(getChunkSize());
                break;
            case SYNC_CHUNK_CODE:
                chunk = new SyncChunk(getChunkSize());
                break;
            case ROIS_CHUNK_CODE:
                chunk = new RoisChunk(getChunkSize());
                break;
            case COMP_CHUNK_CODE:
                chunk = new CompChunk(getChunkSize());
                break;
            case COST_CHUNK_CODE:
                chunk = new CostChunk(getChunkSize());
                break;
            default:
                chunk = nullptr;
        }

        return chunk;
    }
}

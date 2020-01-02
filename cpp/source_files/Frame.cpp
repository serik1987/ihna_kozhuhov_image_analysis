//
// Created by serik1987 on 01.01.2020.
//

#include "Frame.h"

namespace GLOBAL_NAMESPACE{

    Frame::Frame(const FileTrain& train){
        parent = &train;
        iLock = false;
        framChunk = nullptr;
        stimulationChunk = nullptr;
        emode = train.getExperimentalMode();
        body = new uint16_t[train.getXYSize()];
    }

    Frame::~Frame(){
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE FRAME\n";
#endif
        delete framChunk;
        delete stimulationChunk;
        delete [] body;
    }

    std::ostream &operator<<(std::ostream &out, const Frame &frame) {
        out << "===== FRAME # " << frame.getFrameNumber() << " =====\n";
        if (frame.iLock){
            out << "The frame is locked\n";
        }
        if (frame.emode == FileTrain::Continuous){
            out << "Continuous experiment mode\n";
        } else if (frame.emode == FileTrain::Episodic){
            out << "Episodic experiment mode\n";
        }
        if (frame.framChunk == nullptr){
            out << "No frame was loaded\n";
        } else {
            out << *frame.framChunk << "\n";
        }
        if (frame.stimulationChunk != nullptr){
            out << *frame.stimulationChunk << "\n";
        }

        for (int idx = 0; idx < 5; ++idx){
            out << frame.body[idx] << "\t";
        }
        out << "\n";

        return out;
    }

    void Frame::readFromFile(SourceFile &file, int n) {
        delete framChunk;
        framChunk = nullptr;
        delete stimulationChunk;
        stimulationChunk = nullptr;
        frameNumber = n;
        ChunkHeader framChunkHeader = file.readChunkHeader();
        if (framChunkHeader != ChunkHeader::FRAM_CHUNK_CODE){
            throw fram_chunk_not_found_exception(parent, n);
        }
        framChunk = (FramChunk*)framChunkHeader.createChunk();
        framChunk->readFromFile(file);
        ChunkHeader stimulationChunkHeader = file.readChunkHeader();
        if (emode == FileTrain::Continuous && stimulationChunkHeader != ChunkHeader::cost_CHUNK_CODE){
            throw FileTrain::experimental_chunk_not_found(&file);
        }
        if (emode == FileTrain::Episodic && stimulationChunkHeader != ChunkHeader::epst_CHUNK_CODE){
            throw FileTrain::experimental_chunk_not_found(&file);
        }
        stimulationChunk = stimulationChunkHeader.createChunk();
        stimulationChunk->readFromFile(file);
        auto* buffer = (char*)body;
        size_t body_size = parent->getFrameImageSize();
        file.getFileStream().read(buffer, body_size);
    }

}

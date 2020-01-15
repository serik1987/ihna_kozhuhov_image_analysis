//
// Created by serik1987 on 11.01.2020.
//

#include "PixelListItem.h"
#include "TraceReader.h"

namespace GLOBAL_NAMESPACE{

    PixelListItem::PixelListItem(const TraceReader& reader, int row, int col){
        if (row == ARRIVAL_TIME){
            pointType = ArrivalTime;
            i = 0;
            j = 0;
            displacement = reader.getArrivalTimeDisplacement();
        } else if (row == SYNCH){
            if (col < 0 || col >= reader.getSynchChannelNumber()){
                throw TraceReader::TraceNameException();
            }
            pointType = SynchronizationChannel;
            i = 0;
            j = col;
            displacement = reader.getSynchronizationChannelDisplacement() + POINT_SIZE[SynchronizationChannel] * col;
        } else {
            if (row < 0 || col < 0 || row >= reader.getMapSizeY() || col >= reader.getMapSizeX()){
                throw TraceReader::TraceNameException();
            }
            pointType = PixelValue;
            i = row;
            j = col;
            displacement = reader.getFrameBodyDisplacement() +
                    POINT_SIZE[PixelValue] * (i * reader.getMapSizeX() + j);
        }
    }

    PixelListItem::PixelListItem(const PixelListItem& other){
        pointType = other.pointType;
        i = other.i;
        j = other.j;
        displacement = other.displacement;
    }

    PixelListItem& PixelListItem::operator=(const PixelListItem& other){
        pointType = other.pointType;
        i = other.i;
        j = other.j;
        displacement = other.displacement;
        return *this;
    }

    std::ostream& operator<<(std::ostream& out, const PixelListItem& other){
        using std::endl, std::hex, std::dec;

        if (other.pointType == PixelListItem::ArrivalTime){
            out << "Channel type: arrival time\n";
        }

        if (other.pointType == PixelListItem::SynchronizationChannel){
            out << "Channel type: synchronization channel\n";
            out << "Channel number: " << other.j << endl;
        }

        if (other.pointType == PixelListItem::PixelValue){
            out << "Channel type: pixel value\n";
            out << "Row: " << other.i << endl;
            out << "Col: " << other.j << endl;
        }

        out << "Displacement: 0x" << hex << other.getDisplacement() << dec << endl;
        out << "Size: " << other.getPointSize() << endl;
        out << "---------------------------------------------\n";

        return out;
    }

}
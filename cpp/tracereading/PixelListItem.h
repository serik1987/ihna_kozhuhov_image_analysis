//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PIXELLISTITEM_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PIXELLISTITEM_H

#include "../exceptions.h"

namespace GLOBAL_NAMESPACE {

    class TraceReader;

    class PixelListItem {
    public:
        enum PointType {ArrivalTime = 0, SynchronizationChannel = 1, PixelValue = 2};
        static constexpr int ARRIVAL_TIME = -1;
        static constexpr int SYNCH = -2;

        static constexpr int POINT_SIZE[] = {sizeof(uint64_t), sizeof(uint32_t), sizeof(uint16_t)};

    private:
        int i, j;
        size_t displacement;
        PointType pointType;

    public:
        PixelListItem(const TraceReader& reader, int row, int col);
        PixelListItem(const PixelListItem& other);
        PixelListItem& operator=(const PixelListItem& other);

        friend std::ostream& operator<<(std::ostream& out, const PixelListItem& item);

        [[nodiscard]] size_t getDisplacement() const { return displacement; }
        [[nodiscard]] int getPointSize() const { return POINT_SIZE[pointType]; }
        [[nodiscard]] int getI() const { return i; }
        [[nodiscard]] int getJ() const { return j; }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PIXELLISTITEM_H

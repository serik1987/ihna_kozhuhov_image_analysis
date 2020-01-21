//
// Created by serik1987 on 12.01.2020.
//

#include <cmath>
#include <vector>
#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    double dotprod(const std::vector<double>& a, const std::vector<double>& b, int n){
        int i;
        double dumd;
        for(dumd=0.0,i=0;i<n;i++) dumd+=a[i]*b[i];
        return(dumd);
    }

    double PAP(const std::vector<double>& p, int n, std::vector<double>& Ap, std::vector<double>& up,
            const std::vector<double>& vc, const std::vector<double>& vs){
        double pap,dumd,co,si,coN,siN,dumd1,dc,ds;
        int i,j;

        for(i=0,dumd=p[0];i<n/2;i++) dumd+=p[2*i+1];
        up[0]=dumd;
        for(j=1;j<n;j++){
            co=vc[j];
            si=vs[j];
            coN=1.0;
            siN=0.0;
            for(i=0,dumd=p[0];i<(n-1)/2;i++){
                dumd1=coN*co-siN*si;
                siN=siN*co+coN*si;
                coN=dumd1;
                dumd+=p[2*i+1]*coN+p[2*(i+1)]*siN;
            }
            if(!(n%2)){
                dumd+=p[n-1]*(coN*co-siN*si);
            }
            up[j]=dumd;
        }
        pap=dotprod(up,up,n);

        for(i=1,dumd=up[0];i<n;i++) dumd+=up[i];
        Ap[0]=dumd;
        Ap[1]=dotprod(up,vc,n);
        Ap[2]=dotprod(up,vs,n);

        for(j=1;j<(n-1)/2;j++){
            dc=up[0];
            ds=0.0;
            co=vc[j+1];
            si=vs[j+1];
            coN=1.0;
            siN=0.0;
            for(i=1;i<n;i++){
                dumd1=coN*co-siN*si;
                siN=siN*co+coN*si;
                coN=dumd1;
                dc+=up[i]*coN;
                ds+=up[i]*siN;
            }
            Ap[2*j+1]=dc;
            Ap[2*j+2]=ds;
        }
        if(!(n%2)){
            dc=up[0];
            co=vc[n/2];
            si=vs[n/2];
            coN=1.0;
            siN=0.0;
            for(i=1;i<n;i++){
                dumd1=coN*co-siN*si;
                siN=siN*co+coN*si;
                coN=dumd1;
                dc+=up[i]*coN;
            }
            Ap[n-1]=dc;
        }

        return(pap);
    }

    Synchronization::Synchronization(StreamFileTrain &train): train(train) {
        doPrecise = false;
        synchronized = false;
        referenceSignalCos = nullptr;
        referenceSignalSin = nullptr;
        harmonic = 1.0;
        initialFrame = -1;
        finalFrame = -1;
        synchronizationPhase = nullptr;
        phaseIncrement = 0.0;
        initialPhase = 0.0;
        progressFunction = nullptr;
        handle = nullptr;

        if (!train.isOpened()){
            throw FileNotOpenedException();
        }
    }

    Synchronization::~Synchronization(){
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE SYNCHRONIZATION\n";
#endif
        delete [] referenceSignalCos;
        delete [] referenceSignalSin;
        delete [] synchronizationPhase;
    }

    Synchronization::Synchronization(Synchronization&& other) noexcept: train(other.train){
        doPrecise = other.doPrecise;
        synchronized = other.synchronized;
        referenceSignalCos = other.referenceSignalCos;
        other.referenceSignalCos = nullptr;
        referenceSignalSin = other.referenceSignalSin;
        other.referenceSignalSin = nullptr;
        harmonic = other.harmonic;
        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;
        synchronizationPhase = other.synchronizationPhase;
        other.synchronizationPhase = nullptr;
        phaseIncrement = other.phaseIncrement;
        initialPhase = other.initialPhase;
        progressFunction = other.progressFunction;
        handle = other.handle;
    }

    Synchronization& Synchronization::operator=(Synchronization&& other) noexcept{
        doPrecise = other.doPrecise;
        synchronized = other.synchronized;
        referenceSignalCos = other.referenceSignalCos;
        other.referenceSignalCos = nullptr;
        referenceSignalSin = other.referenceSignalSin;
        other.referenceSignalSin = nullptr;
        harmonic = other.harmonic;

        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;
        train = other.train;
        synchronizationPhase = other.synchronizationPhase;
        other.synchronizationPhase = nullptr;
        phaseIncrement = other.phaseIncrement;
        initialPhase = other.initialPhase;

        progressFunction = other.progressFunction;
        handle = other.handle;

        return *this;
    }

    const double *Synchronization::getSynchronizationPhase() const {
        if (!synchronized || synchronizationPhase  == nullptr){
            throw NotSynchronizedException();
        } else {
            return synchronizationPhase;
        }
    }

    const double *Synchronization::getReferenceSignalCos() const {
        if (!synchronized || referenceSignalCos == nullptr){
            throw NotSynchronizedException();
        } else {
            return referenceSignalCos;
        }
    }

    const double *Synchronization::getReferenceSignalSin() const {
        if (!synchronized || referenceSignalSin == nullptr){
            throw NotSynchronizedException();
        } else {
            return referenceSignalSin;
        }
    }

    std::ostream &operator<<(std::ostream &out, const Synchronization &sync) {
        out << "===== SYNCHRONIZATION =====\n";
        out << "Synchronization type: " << sync.getName() << "\n";
        out << "Initial frame: " << sync.getInitialFrame() << "\n";
        out << "Final frame: " << sync.getFinalFrame() << "\n";
        out << "Frame number: " << sync.getFrameNumber() << "\n";
        if (sync.isDoPrecise()) {
            out << "Precise analysis is ON\n";
        } else {
            out << "Precise analysis is OFF\n";
        }
        out << "Harmonic: " << sync.getHarmonic() << "\n";
        if (sync.isSynchronized()){
            out << "Synchronization is completed\n";
            out << "Phase increment, rad: " << sync.getPhaseIncrement() << "\n";
            out << "Initial phase, rad: " << sync.getInitialPhase() << "\n";

        } else {
            out << "Synchronization is not completed\n";
        }

        sync.specialPrint(out);

        return out;
    }

    void Synchronization::synchronize() {
        printf("\n");
        clearState();
        calculateSynchronizationPhase();
        calculatePhaseIncrement();
        inverse();

        synchronized = true;
    }

    void Synchronization::clearState() {
        synchronized = false;
        delete [] referenceSignalCos;
        delete [] referenceSignalSin;
        delete [] synchronizationPhase;
    }

    void Synchronization::inverse() {
        referenceSignalCos = new double[getFrameNumber()];
        referenceSignalSin = new double[getFrameNumber()];
        double h = getCycleNumber() * getHarmonic();

        if (isDoPrecise()){
            inversePrecise(h);
        } else {
            referenceSignalCos[0] = 1.0;
            referenceSignalSin[0] = 0.0;
            referenceSignalCos[1] = cos(phaseIncrement * h);
            referenceSignalSin[1] = sin(phaseIncrement * h);
            for (int i=2; i < getFrameNumber(); ++i){
                referenceSignalCos[i] = referenceSignalCos[i-1] * referenceSignalCos[1]
                        - referenceSignalSin[i-1] * referenceSignalSin[1];
                referenceSignalSin[i] = referenceSignalSin[i-1] * referenceSignalCos[1]
                        + referenceSignalCos[i-1] * referenceSignalSin[1];
            }
            double A = referenceSignalCos[0] = 2.0 / (double)getFrameNumber();
            for (int i=1; i < getFrameNumber(); ++i){
                referenceSignalCos[i] *= A;
                referenceSignalSin[i] *= A;
            }
        }
    }

    void Synchronization::inversePrecise(double h) {
        using std::vector;

        int k=2*(int)h-1;
        int n = getFrameNumber();
        double omega = phaseIncrement;

        if (k < 1 || k >= n){
            throw BadHarmonicException();
        }

        int it=(int)(ITERATIONM*(float)n);
        int o=n;
        int print_p = 10;
        if(n>10000) print_p=2;

        vector<double> vc(n);
        vector<double> vs(n);
        vector<double> sol[] = {vector<double>(n), vector<double>(n)};
        vector<double> up(n);
        vector<double> r(n);
        vector<double> p(n);
        vector<double> Ap(n);

        vc[0]=1.0;
        vs[0]=0.0;
        vc[1]=cos(omega);
        vs[1]=sin(omega);
        for(int i=2;i<n;i++){
            vc[i]=vc[i-1]*vc[1]-vs[i-1]*vs[1];
            vs[i]=vs[i-1]*vc[1]+vc[i-1]*vs[1];
        }

        for(int m=0;m<2;m++){
            vector<double>& sol0=sol[m];
            for(int i=0;i<n;i++) sol0[i]=0.0;
            sol0[k]=2.0/(double)n;
            PAP(sol[m],n,Ap,up,vc,vs);
            for(int i=0;i<n;i++) r[i]=p[i]=-Ap[i];
            r[k]+=1.0;
            p[k]+=1.0;

            int l=0;
            double rr=dotprod(r,r,n);
            while(1){
                double a=rr/PAP(p,n,Ap,up,vc,vs);
                for(int j=0;j<n;j++) r[j]-=a*Ap[j];
                double dumd;
                double b=(dumd=dotprod(r,r,n))/rr;
                rr=dumd;
                for(int j=0;j<n;j++){
                    sol0[j]+=a*p[j];
                    p[j]=r[j]+b*p[j];
                }
                if(l>it-10) printf("INV %i %f\n",l,rr);
                if(rr<EPSILON || l==it){
                    printf("INV It=%i, rr=%e\n",l,rr);
                    break;
                }
                l++;
                if(!(l%print_p)) printf("INV %i %e\n",l,rr);
            }
            printf("INV First %e\n",sol0[0]);

            double maxy[2], miny[2], maxy2[2];
            int imax[2], imax2[2], imin[2];
            if(fabs(sol0[0])>fabs(sol0[1])){
                maxy[m]=fabs(sol0[0]);
                imax[m]=0;
                miny[m]=maxy2[m]=fabs(sol0[1]);
                imax2[m]=imin[m]=1;
            }
            else{
                miny[m]=maxy2[m]=fabs(sol0[0]);
                imax2[m]=imin[m]=0;
                maxy[m]=fabs(sol0[1]);
                imax[m]=1;
            }
            for(int i=2;i<n;i++){
                double dumd=fabs(sol0[i]);
                if(miny[m]>dumd){
                    miny[m]=dumd;
                    imin[m]=i;
                }
                if(maxy2[m]<dumd){
                    maxy2[m]=dumd;
                    imax2[m]=i;
                    if(maxy[m]<maxy2[m]){
                        dumd=maxy[m];
                        maxy[m]=maxy2[m];
                        maxy2[m]=dumd;
                        l=imax[m];
                        imax[m]=imax2[m];
                        imax2[m]=l;
                    }
                }
            }

            printf("INV %i MAX %e(%e)(%i) NMAX %e(%i) MIN %e(%i)\n",
                    k, maxy[m], sol[m][imax[m]], imax[m], maxy2[m], imax2[m], miny[m], imin[m]);
            k++;
        }

        vector<double>& sol0=sol[0];
        vector<double>& sol1=sol[1];

        double alpha = initialPhase;
        double coa=cos(alpha);
        double sia=sin(alpha);
        double dc;
        double ds;
        int i;

        for(i=0,dc=sol0[0],ds=sol1[0];i<n/2;i++){
            dc+=sol0[2*i+1];
            ds+=sol1[2*i+1];
        }
        referenceSignalCos[0]=dc*coa-ds*sia;
        referenceSignalSin[0]=ds*coa+dc*sia;
        for (int j=1;j<n;j++){
            double co=vc[j];
            double si=vs[j];
            double coN=1.0;
            double siN=0.0;
            for(i=0,dc=sol0[0],ds=sol1[0];i<(n-1)/2;i++){
                double dumd=coN*co-siN*si;
                siN=siN*co+coN*si;
                coN=dumd;
                dc+=sol0[2*i+1]*coN+sol0[2*(i+1)]*siN;
                ds+=sol1[2*i+1]*coN+sol1[2*(i+1)]*siN;
            }
            if(!(n%2)){
                dc+=sol0[n-1]*(coN*co-siN*si);
                ds+=sol1[n-1]*(coN*co-siN*si);
            }
            referenceSignalCos[j]=dc*coa-ds*sia;
            referenceSignalSin[j]=ds*coa+dc*sia;
        }
    }
}
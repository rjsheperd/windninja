/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Command line wrapper around momentum_compare
 *
 ******************************************************************************
 *
 * THIS SOFTWARE WAS DEVELOPED AT THE ROCKY MOUNTAIN RESEARCH STATION (RMRS)
 * MISSOULA FIRE SCIENCES LABORATORY BY EMPLOYEES OF THE FEDERAL GOVERNMENT
 * IN THE COURSE OF THEIR OFFICIAL DUTIES. PURSUANT TO TITLE 17 SECTION 105
 * OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO COPYRIGHT
 * PROTECTION AND IS IN THE PUBLIC DOMAIN. RMRS MISSOULA FIRE SCIENCES
 * LABORATORY ASSUMES NO RESPONSIBILITY WHATSOEVER FOR ITS USE BY OTHER
 * PARTIES,  AND MAKES NO GUARANTEES, EXPRESSED OR IMPLIED, ABOUT ITS QUALITY,
 * RELIABILITY, OR ANY OTHER CHARACTERISTIC.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

/*
** usage: compare_momentum CASE_DIR REFERENCE_DIR
**                         [MAX_SPEED_RMSE MAX_DIRECTION_DIFF MAX_K_REL_RMSE]
**
** Prints the momentum_compare metrics.  With limits given, exits 1 if any
** metric exceeds its limit.  Exits 2 on bad usage or unreadable input.
*/

#include <cstdio>
#include <cstdlib>
#include <exception>

#include "momentum_compare.h"

int main(int argc, char *argv[])
{
    if(argc != 3 && argc != 6)
    {
        fprintf(stderr, "usage: %s CASE_DIR REFERENCE_DIR "
                        "[MAX_SPEED_RMSE MAX_DIRECTION_DIFF MAX_K_REL_RMSE]\n",
                argv[0]);
        return 2;
    }

    MomentumMetrics m;
    try
    {
        m = CompareMomentumCases(argv[1], argv[2]);
    }
    catch(std::exception &e)
    {
        fprintf(stderr, "compare_momentum: %s\n", e.what());
        return 2;
    }

    printf("cells:               %d\n", m.nCells);
    printf("speed RMSE:          %.6g m/s\n", m.speedRmse);
    printf("speed max |diff|:    %.6g m/s\n", m.speedMaxAbs);
    printf("direction mean diff: %.6g deg\n", m.directionMeanAbs);
    printf("k relative RMSE:     %.6g\n", m.kRelativeRmse);

    if(argc == 6)
    {
        bool pass = m.speedRmse <= atof(argv[3]) &&
                    m.directionMeanAbs <= atof(argv[4]) &&
                    m.kRelativeRmse <= atof(argv[5]);
        printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    return 0;
}

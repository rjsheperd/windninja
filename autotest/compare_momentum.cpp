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
**
** usage: compare_momentum --relative CASE_DIR SAME_MESH_DIR FINE_DIR
**
** The terrain criterion (WEBGPU_PLAN GATE 2, user decision 2026-10-21): CASE
** and SAME_MESH (OpenFOAM at the same mesh) are both compared with FINE
** (OpenFOAM's fine mesh) at the 20 ft output, resampled onto CASE's cells.
** Passes (exit 0) if CASE is no worse than SAME_MESH in speed RMSE and in
** mean direction difference.
*/

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "momentum_compare.h"

static void Print(const char *label, const MomentumMetrics &m)
{
    printf("%-18s cells %d, speed RMSE %.4g m/s, max %.4g, direction %.4g deg\n",
           label, m.nCells, m.speedRmse, m.speedMaxAbs, m.directionMeanAbs);
}

static int Relative(const char *caseDir, const char *sameMesh, const char *fine)
{
    MomentumMetrics ours, theirs;
    try
    {
        ours = CompareMomentumCasesResampled(caseDir, fine);
        theirs = CompareMomentumCasesResampled(sameMesh, fine);
    }
    catch(std::exception &e)
    {
        fprintf(stderr, "compare_momentum: %s\n", e.what());
        return 2;
    }
    printf("against the fine reference (20 ft):\n");
    Print("  case", ours);
    Print("  same-mesh ref", theirs);
    bool pass = ours.speedRmse <= theirs.speedRmse &&
                ours.directionMeanAbs <= theirs.directionMeanAbs;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char *argv[])
{
    if(argc == 5 && std::string(argv[1]) == "--relative")
        return Relative(argv[2], argv[3], argv[4]);
    if(argc != 3 && argc != 6)
    {
        fprintf(stderr, "usage: %s CASE_DIR REFERENCE_DIR "
                        "[MAX_SPEED_RMSE MAX_DIRECTION_DIFF MAX_K_REL_RMSE]\n"
                        "       %s --relative CASE_DIR SAME_MESH_DIR FINE_DIR\n",
                argv[0], argv[0]);
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

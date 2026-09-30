from market.universe import SYMBOLS

from market.scanner import (
    MarketScanner
)

from strategy.ranker import (
    CandidateRanker
)

from strategy.allocation import (
    CapitalAllocator
)

from config.settings import (
    TRADING_CAPITAL,
    MAX_CAPITAL_DEPLOYED,
)

def print_scan_results(results):

    print()
    print("=" * 82)
    print("RAW MARKET SCAN")
    print("=" * 82)

    print(
        f"{'SYMBOL':<8}"
        f"{'LONG':>10}"
        f"{'SHORT':>10}"
        f"{'DIR':>10}"
        f"{'5M':>10}"
        f"{'RVOL':>10}"
        f"{'ACTIVE':>10}"
    )

    print("-" * 82)

    for item in results:

        print(
            f"{item['symbol']:<8}"
            f"{item['long_probability']:>10.2%}"
            f"{item['short_probability']:>10.2%}"
            f"{item['direction']:>10}"
            f"{item['return_5m']:>10.2%}"
            f"{item['relative_volume']:>10.2f}"
            f"{str(item['active']):>10}"
        )

def print_rankings(ranked):

    print()
    print("=" * 82)
    print("TOP OPPORTUNITIES")
    print("=" * 82)

    print(
        f"{'#':<4}"
        f"{'SYMBOL':<8}"
        f"{'DIR':<9}"
        f"{'LONG':>10}"
        f"{'SHORT':>10}"
        f"{'SCORE':>10}"
        f"{'5M':>10}"
        f"{'RVOL':>10}"
    )

    print("-" * 82)

    for index, item in enumerate(ranked, start=1):

        print(
            f"{index:<4}"
            f"{item['symbol']:<8}"
            f"{item['direction']:<9}"
            f"{item['long_probability']:>10.2%}"
            f"{item['short_probability']:>10.2%}"
            f"{item['score']:>10.3f}"
            f"{item['return_5m']:>10.2%}"
            f"{item['relative_volume']:>10.2f}"
        )

def print_allocation(allocation):

    print()
    print("=" * 82)
    print("AI CAPITAL ALLOCATION")
    print("=" * 82)

    print(
        f"Defined capital:        "
        f"${TRADING_CAPITAL:,.2f}"
    )

    print(
        f"Maximum deployment:     "
        f"${TRADING_CAPITAL * MAX_CAPITAL_DEPLOYED:,.2f}"
    )

    print()

    positions = allocation["positions"]

    if not positions:

        print("NO POSITIONS SELECTED.")
        print()

        print(
            f"Cash reserved:          "
            f"${allocation['cash_reserved']:,.2f}"
        )

        return

    print(
        f"{'SYMBOL':<8}"
        f"{'DIR':<9}"
        f"{'LONG':>10}"
        f"{'SHORT':>10}"
        f"{'SCORE':>10}"
        f"{'ALLOC':>14}"
    )

    print("-" * 82)

    for position in positions:

        allocation_text = (
            f"${position['allocation']:.2f}"
        )

        print(
            f"{position['symbol']:<8}"
            f"{position['direction']:<9}"
            f"{position['long_probability']:>10.2%}"
            f"{position['short_probability']:>10.2%}"
            f"{position['score']:>10.3f}"
            f"{allocation_text:>14}"
        )

    print()

    print(
        f"Capital deployed:       "
        f"${allocation['capital_deployed']:,.2f}"
    )

    print(
        f"Cash reserved:          "
        f"${allocation['cash_reserved']:,.2f}"
    )

def main():

    print()
    print(
        "=" * 76
    )

    print(
        "AI QUICK-TRADING SCANNER"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Scanning "
        f"{len(SYMBOLS)} symbols..."
    )


    # ==================================
    # 1. MARKET SCANNER
    # ==================================

    scanner = MarketScanner()


    results = scanner.scan_universe(
        SYMBOLS
    )


    if len(results) == 0:

        print()
        print(
            "ERROR: Scanner returned "
            "no usable stocks."
        )

        return


    print_scan_results(
        results
    )


    # ==================================
    # 2. RANK OPPORTUNITIES
    # ==================================

    ranker = CandidateRanker()


    ranked = ranker.rank(
        results
    )


    if len(ranked) == 0:

        print()

        print(
            "No stocks passed the "
            "activity scanner."
        )

        return


    print_rankings(
        ranked
    )


    # ==================================
    # 3. CAPITAL ALLOCATION
    # ==================================

    allocator = (
        CapitalAllocator()
    )


    allocation = (
        allocator.allocate(
            ranked
        )
    )


    print_allocation(
        allocation
    )


    # ==================================
    # 4. SAFETY
    # ==================================

    print()
    print(
        "=" * 76
    )

    print(
        "PAPER MODE / ANALYSIS ONLY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "No trades were submitted."
    )

    print(
        "The allocations above are "
        "candidate allocations only."
    )


if __name__ == "__main__":

    main()
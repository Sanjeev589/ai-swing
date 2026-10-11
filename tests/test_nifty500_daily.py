import pandas as pd
import pytest
from data.nifty500_daily import parse_bhavcopy, validate_ocv

CSV = '''SYMBOL, SERIES, DATE1, OPEN_PRICE, CLOSE_PRICE, TTL_TRD_QNTY
AAA, EQ, 11-Oct-2019, 100, 101.25, 5000
BBB, BE, 11-Oct-2019, 50, 51, 1000
CCC, EQ, 11-Oct-2019, 70, 72, 2000
'''


def test_official_daily_schema():
    result = parse_bhavcopy(CSV, '2019-10-11', {'AAA', 'BBB'})
    assert list(result.columns) == ['date', 'symbol', 'open', 'close', 'volume']
    assert result.symbol.tolist() == ['AAA', 'BBB']
    assert result.iloc[0].volume == 5000
    assert result.iloc[0].close == 101.25
    with pytest.raises(ValueError, match='does not match'):
        parse_bhavcopy(CSV, '2019-10-12', {'AAA'})


def test_no_invalid_or_fabricated_rows():
    frame = parse_bhavcopy(CSV, '2019-10-11', {'AAA'})
    with pytest.raises(ValueError, match='duplicate'):
        validate_ocv(pd.concat([frame, frame]))
    frame.loc[0, 'open'] = 0
    with pytest.raises(ValueError, match='Invalid prices'):
        validate_ocv(frame)
    with pytest.raises(ValueError):
        parse_bhavcopy(CSV, '2019-10-11', {'ABSENT'})


def test_download_end_to_end_with_audits(tmp_path, monkeypatch):
    import json
    from data.nifty500_daily import download_dataset
    snapshot = pd.DataFrame({'Symbol': ['AAA'] + [f'S{i}' for i in range(499)],
                             'ISIN Code': [f'ISIN{i}' for i in range(500)]})
    source = tmp_path / 'constituents.csv'
    snapshot.to_csv(source, index=False)
    class Response:
        status_code = 200
        text = CSV
        content = CSV.encode()
        def raise_for_status(self):
            pass
    monkeypatch.setattr('data.nifty500_daily.requests.get', lambda *a, **k: Response())
    output = tmp_path / 'dataset'
    manifest = download_dataset('2019-10-11', '2019-10-12', output, workers=1,
                                constituents_file=source, cache=tmp_path / 'cache')
    assert manifest['rows'] == 1
    assert manifest['constituents'] == 500
    stored = pd.read_csv(output / 'daily_2019.csv.gz')
    assert stored.volume.tolist() == [5000]
    coverage = pd.read_csv(output / 'coverage.csv')
    assert coverage.rows.sum() == 1
    assert (coverage.rows == 0).sum() == 499
    assert json.loads((output / 'manifest.json').read_text())['download_errors'] == 0


def test_mislabeled_excel_archive():
    from io import BytesIO, StringIO
    workbook = BytesIO()
    pd.read_csv(StringIO(CSV), skipinitialspace=True).to_excel(workbook, index=False)
    parsed = parse_bhavcopy(workbook.getvalue(), '2019-10-11', {'AAA'})
    assert parsed.iloc[0].volume == 5000
    assert parsed.iloc[0].close == 101.25


def test_reit_daily_records():
    text = CSV + "EMBASSY, RR, 11-Oct-2019, 425, 425.09, 697000\n"
    result = parse_bhavcopy(text, '2019-10-11', {'EMBASSY'})
    assert result.symbol.tolist() == ['EMBASSY']
    assert result.iloc[0].volume == 697000

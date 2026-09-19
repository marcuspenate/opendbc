from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.can_definitions import CanData
from opendbc.car.hyundai.hyundaican import DAW_REST_RECOMMEND, create_lkas12
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR

LKAS12_ADDR = 0x53E
LKAS11_ADDR = 0x340
DBC = "hyundai_can_generated"

# camera LKAS11 captured during a popup: bit 15 (CF_Lkas_FcwBasReq) is the cluster's chime request on this car
CAMERA_LKAS11_CHIME = bytes.fromhex("fc83002476004f04")
CHIME_BIT = 1 << 15

# exact frame the 2025 Elantra Hybrid Limited camera sends: 8 bytes, DAW level 5, several bits the DBC doesn't name
CAMERA_FRAME = bytes.fromhex("0200000008051400")


def _daw(dat: bytes) -> int:
  return (int.from_bytes(dat, "little") >> 40) & 0x7


REST_RECOMMEND_BIT = 1 << 44   # raised by the camera together with status 6; the cluster keys on it


def _with_daw(dat: bytes, daw: int) -> bytes:
  v = int.from_bytes(dat, "little") & ~(0x7 << 40) | (daw << 40)
  return v.to_bytes(8, "little")


def _popup(dat: bytes) -> bytes:
  """The frame the camera actually sends during "Consider taking a break": status 6 plus bit 44."""
  v = int.from_bytes(_with_daw(dat, DAW_REST_RECOMMEND), "little") | REST_RECOMMEND_BIT
  return v.to_bytes(8, "little")


def _make_interface() -> CarInterface:
  cand = CAR.HYUNDAI_ELANTRA_HEV_2024
  CP = CarInterface.get_non_essential_params(cand)
  CI = CarInterface(CP, CarInterface.get_non_essential_params_sp(CP, cand))
  CI.update([(0, [])])  # first parser update is a no-op for vl_all, as on the device it runs at 100 Hz
  return CI


def _parse_camera(dat: bytes) -> dict:
  """What CarState holds for the camera's LKAS12 after it has been received once."""
  CI = _make_interface()
  CI.update([(10_000_000, [CanData(LKAS12_ADDR, dat, 2)])])
  return dict(CI.CS.lkas12)


class TestLkas12Daw:
  def setup_method(self):
    self.packer = CANPacker(DBC)

  def test_real_frame_round_trips_bit_exact(self):
    for daw in (0, 1, 2, 3, 4, 5, 7):
      cam = _with_daw(CAMERA_FRAME, daw)
      addr, dat, bus = create_lkas12(self.packer, _parse_camera(cam), daw_level=3)
      assert (addr, bus) == (LKAS12_ADDR, 0)
      assert dat == cam, f"daw={daw}: sent {dat.hex()} != camera {cam.hex()}"

  def test_popup_is_replaced_with_last_level_and_flag_cleared(self):
    for cam in (_with_daw(CAMERA_FRAME, DAW_REST_RECOMMEND), _popup(CAMERA_FRAME)):
      for level in range(1, 6):
        _, dat, _ = create_lkas12(self.packer, _parse_camera(cam), daw_level=level)
        assert _daw(dat) == level
        assert not int.from_bytes(dat, "little") & REST_RECOMMEND_BIT
        assert dat == _with_daw(CAMERA_FRAME, level)   # every other bit untouched

  def test_length_matches_camera(self):
    _, dat, _ = create_lkas12(self.packer, _parse_camera(CAMERA_FRAME), daw_level=5)
    assert len(dat) == len(CAMERA_FRAME) == 8


class TestLkas12DawEndToEnd:
  def setup_method(self):
    self.CI = _make_interface()
    self.CC = structs.CarControl().as_reader()
    self.CC_SP = structs.CarControlSP()  # sunnypilot dataclass, not a capnp struct
    self.frame = 0
    self.t = 0

  def _camera(self, daw: int):
    self.t += 10_000_000
    self.CI.update([(self.t, [CanData(LKAS12_ADDR, _with_daw(CAMERA_FRAME, daw), 2)])])

  def _sent(self, frames: int = 20, addr: int = LKAS12_ADDR) -> list:
    out = []
    for _ in range(frames):
      out += [m for m in self.CI.apply(self.CC, self.CC_SP, self.frame * 10_000_000)[1] if m[0] == addr]
      self.frame += 1
    return out

  def _chime_sent(self) -> list[bool]:
    """Feed the camera's chime-requesting LKAS11 and report whether each LKAS11 we send still carries the bit."""
    self.t += 10_000_000
    self.CI.update([(self.t, [CanData(LKAS11_ADDR, CAMERA_LKAS11_CHIME, 2)])])
    assert self.CI.CS.lkas11["CF_Lkas_FcwBasReq"] == 1
    return [bool(int.from_bytes(m[1], "little") & CHIME_BIT) for m in self._sent(addr=LKAS11_ADDR)]

  def test_nothing_sent_until_camera_produces_lkas12(self):
    assert not self.CI.CS.lkas12_seen
    assert self._sent() == []

  def test_level_passes_through_and_popup_holds_last_level(self):
    self._camera(4)
    assert self.CI.CS.lkas12_seen
    sent = self._sent()
    assert len(sent) == 2                                  # 10 Hz over 20 control frames at 100 Hz
    assert all(m[2] == 0 for m in sent)                    # to the car, not the camera
    assert all(m[1] == _with_daw(CAMERA_FRAME, 4) for m in sent)

    self._camera(1)
    assert all(m[1] == _with_daw(CAMERA_FRAME, 1) for m in self._sent())

    self.t += 10_000_000                                   # popup as the camera really sends it: status 6 + bit 44
    self.CI.update([(self.t, [CanData(LKAS12_ADDR, _popup(CAMERA_FRAME), 2)])])
    assert all(m[1] == _with_daw(CAMERA_FRAME, 1) for m in self._sent())   # masked with the last real level, flag cleared

    self._camera(2)                                        # camera recovers: real level again
    assert all(m[1] == _with_daw(CAMERA_FRAME, 2) for m in self._sent())

  def test_chime_request_masked_only_at_level_1_and_popup(self):
    self._camera(5)                                        # attentive: the bit is none of our business
    assert all(self._chime_sent())

    self._camera(2)
    assert all(self._chime_sent())

    self._camera(1)                                        # popup can follow at any moment: mask armed
    assert not any(self._chime_sent())

    self.t += 10_000_000                                   # the popup itself
    self.CI.update([(self.t, [CanData(LKAS12_ADDR, _popup(CAMERA_FRAME), 2)])])
    assert not any(self._chime_sent())

    self._camera(3)                                        # recovered: passes through again
    assert all(self._chime_sent())

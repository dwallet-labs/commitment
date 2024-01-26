// Author: dWallet Labs, LTD.
// SPDX-License-Identifier: BSD-3-Clause-Clear

use std::ops::Mul;

use group::{self_product, BoundedGroupElement, Samplable};
use serde::{Deserialize, Serialize};

use crate::{pedersen, GroupsPublicParameters, HomomorphicCommitmentScheme, Pedersen};

/// A Batched Pedersen Commitment:
/// $$\Com_\pp(m;\rho):=\Ped.\Com_{\GG,G,H,q}(\vec{m},\vec{\rho})=(m_1\cdot G + \rho_1 \cdot H, \ldots, m_n\cdot G + \rho_n \cdot H)$$
#[derive(PartialEq, Clone, Debug, Eq)]
pub struct MultiPedersen<
    const BATCH_SIZE: usize,
    const SCALAR_LIMBS: usize,
    Scalar: group::GroupElement,
    GroupElement: group::GroupElement,
>(Pedersen<1, SCALAR_LIMBS, Scalar, GroupElement>);

impl<const BATCH_SIZE: usize, const SCALAR_LIMBS: usize, Scalar, GroupElement>
    HomomorphicCommitmentScheme<SCALAR_LIMBS>
    for MultiPedersen<BATCH_SIZE, SCALAR_LIMBS, Scalar, GroupElement>
where
    Scalar: BoundedGroupElement<SCALAR_LIMBS>
        + Mul<GroupElement, Output = GroupElement>
        + for<'r> Mul<&'r GroupElement, Output = GroupElement>
        + Samplable
        + Copy,
    GroupElement: group::GroupElement,
{
    type MessageSpaceGroupElement = self_product::GroupElement<BATCH_SIZE, Scalar>;
    type RandomnessSpaceGroupElement = self_product::GroupElement<BATCH_SIZE, Scalar>;
    type CommitmentSpaceGroupElement = self_product::GroupElement<BATCH_SIZE, GroupElement>;
    type PublicParameters = PublicParameters<
        BATCH_SIZE,
        GroupElement::Value,
        Scalar::PublicParameters,
        GroupElement::PublicParameters,
    >;

    fn new(public_parameters: &Self::PublicParameters) -> crate::Result<Self> {
        Pedersen::new(&public_parameters.pedersen_public_parameters).map(Self)
    }

    fn commit(
        &self,
        message: &Self::MessageSpaceGroupElement,
        randomness: &Self::RandomnessSpaceGroupElement,
    ) -> Self::CommitmentSpaceGroupElement {
        // $$\Com_\pp(m;\rho):=\Ped.\Com_{\GG,G,H,q}(\vec{m},\vec{\rho})=(m_1\cdot G + \rho_1 \cdot H, \ldots, m_n\cdot G + \rho_n \cdot H)$$
        let messages: [_; BATCH_SIZE] = (*message).into();
        let randomnesses: [_; BATCH_SIZE] = (*randomness).into();

        let commitments: [_; BATCH_SIZE] = messages
            .into_iter()
            .zip(randomnesses)
            .map(|(message, randomness)| self.0.commit(&[message].into(), &randomness))
            .collect::<Vec<_>>()
            .try_into()
            .ok()
            .unwrap();

        commitments.into()
    }
}

pub type MessageSpaceGroupElement<const BATCH_SIZE: usize, Scalar> =
    self_product::GroupElement<BATCH_SIZE, Scalar>;
pub type MessageSpacePublicParameters<const BATCH_SIZE: usize, Scalar> =
    group::PublicParameters<MessageSpaceGroupElement<BATCH_SIZE, Scalar>>;
pub type RandomnessSpaceGroupElement<Scalar> = Scalar;
pub type RandomnessSpacePublicParameters<Scalar> =
    group::PublicParameters<RandomnessSpaceGroupElement<Scalar>>;
pub type CommitmentSpaceGroupElement<GroupElement> = GroupElement;
pub type CommitmentSpacePublicParameters<GroupElement> =
    group::PublicParameters<CommitmentSpaceGroupElement<GroupElement>>;

/// The Public Parameters of a Pedersen Commitment.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct PublicParameters<
    const BATCH_SIZE: usize,
    GroupElementValue,
    ScalarPublicParameters,
    GroupPublicParameters,
> {
    pub groups_public_parameters: GroupsPublicParameters<
        self_product::PublicParameters<BATCH_SIZE, ScalarPublicParameters>,
        self_product::PublicParameters<BATCH_SIZE, ScalarPublicParameters>,
        self_product::PublicParameters<BATCH_SIZE, GroupPublicParameters>,
    >,
    pub pedersen_public_parameters: pedersen::PublicParameters<
        1,
        GroupElementValue,
        ScalarPublicParameters,
        GroupPublicParameters,
    >,
}

impl<
        const BATCH_SIZE: usize,
        GroupElementValue: Copy,
        ScalarPublicParameters: Clone,
        GroupPublicParameters: Clone,
    >
    From<
        pedersen::PublicParameters<
            1,
            GroupElementValue,
            ScalarPublicParameters,
            GroupPublicParameters,
        >,
    >
    for PublicParameters<
        BATCH_SIZE,
        GroupElementValue,
        ScalarPublicParameters,
        GroupPublicParameters,
    >
{
    fn from(
        pedersen_public_parameters: pedersen::PublicParameters<
            1,
            GroupElementValue,
            ScalarPublicParameters,
            GroupPublicParameters,
        >,
    ) -> Self {
        Self {
            groups_public_parameters: GroupsPublicParameters {
                message_space_public_parameters: self_product::PublicParameters::new(
                    pedersen_public_parameters
                        .groups_public_parameters
                        .randomness_space_public_parameters
                        .clone(),
                ),
                randomness_space_public_parameters: self_product::PublicParameters::new(
                    pedersen_public_parameters
                        .groups_public_parameters
                        .randomness_space_public_parameters
                        .clone(),
                ),
                commitment_space_public_parameters: self_product::PublicParameters::new(
                    pedersen_public_parameters
                        .groups_public_parameters
                        .commitment_space_public_parameters
                        .clone(),
                ),
            },
            pedersen_public_parameters,
        }
    }
}

impl<const BATCH_SIZE: usize, GroupElementValue, ScalarPublicParameters, GroupPublicParameters>
    AsRef<
        GroupsPublicParameters<
            self_product::PublicParameters<BATCH_SIZE, ScalarPublicParameters>,
            self_product::PublicParameters<BATCH_SIZE, ScalarPublicParameters>,
            self_product::PublicParameters<BATCH_SIZE, GroupPublicParameters>,
        >,
    >
    for PublicParameters<
        BATCH_SIZE,
        GroupElementValue,
        ScalarPublicParameters,
        GroupPublicParameters,
    >
{
    fn as_ref(
        &self,
    ) -> &GroupsPublicParameters<
        self_product::PublicParameters<BATCH_SIZE, ScalarPublicParameters>,
        self_product::PublicParameters<BATCH_SIZE, ScalarPublicParameters>,
        self_product::PublicParameters<BATCH_SIZE, GroupPublicParameters>,
    > {
        &self.groups_public_parameters
    }
}

#[cfg(test)]
mod tests {
    use bulletproofs::PedersenGens;
    use group::ristretto;
    use rand_core::OsRng;

    use crate::Pedersen;

    use super::*;

    #[test]
    fn commits() {
        let scalar_public_parameters = ristretto::scalar::PublicParameters::default();
        let group_public_parameters = ristretto::group_element::PublicParameters::default();

        let message = ristretto::Scalar::sample(&scalar_public_parameters, &mut OsRng).unwrap();
        let randomness = ristretto::Scalar::sample(&scalar_public_parameters, &mut OsRng).unwrap();

        let commitment_generators = PedersenGens::default();

        let commitment_scheme_public_parameters = crate::PublicParameters::<
            { ristretto::SCALAR_LIMBS },
            Pedersen<1, { ristretto::SCALAR_LIMBS }, ristretto::Scalar, ristretto::GroupElement>,
        >::new::<
            { ristretto::SCALAR_LIMBS },
            ristretto::Scalar,
            ristretto::GroupElement,
        >(
            scalar_public_parameters,
            group_public_parameters,
            [commitment_generators.B.compress().try_into().unwrap()],
            commitment_generators
                .B_blinding
                .compress()
                .try_into()
                .unwrap(),
        )
        .into();

        let commitment_scheme = MultiPedersen::<
            1,
            { ristretto::SCALAR_LIMBS },
            ristretto::Scalar,
            ristretto::GroupElement,
        >::new(&commitment_scheme_public_parameters)
        .unwrap();

        let expected_commitment = commitment_generators.commit(message.into(), randomness.into());

        let commitment: [_; 1] = commitment_scheme
            .commit(&([message].into()), &([randomness].into()))
            .into();

        assert_eq!(expected_commitment, commitment[0].into())
    }
}
